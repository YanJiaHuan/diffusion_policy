import os
import time
import enum
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from piper_sdk import C_PiperInterface
import serial
from scipy.spatial.transform import Rotation as R


# ============================== #
#           COMMANDS             #
# ============================== #
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_POINT = 3
    MAGNET = 4

# ============================== #
#        PIPER CONTROLLER        #
# ============================== #
class PiperInterpolationController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 can_interface="can_piper",
                 bt_port='/dev/rfcomm0',
                 bt_baud_rate=115200,
                 frequency=10,
                 lookahead_time=0.1,
                 gain=300,
                 max_pos_speed=0.25,
                 max_rot_speed=0.16,
                 launch_timeout=3,
                 tcp_offset_pose=None,
                 payload_mass=None,
                 payload_cog=None,
                 joints_init=None,
                 joints_init_speed=1,
                 reset = True,
                 soft_real_time=False,
                 verbose=False,
                 receive_keys=None,
                 get_max_k=128):

        assert 0<frequency<= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 < payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="PiperInterpolationController")
        self.can_interface = can_interface
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.reset = reset

        #---------------Magnet Controller----------------
        self.bt_port = bt_port
        self.bt_baud_rate = bt_baud_rate
        self.bt_connection = None  # will hold serial.Serial object
        self.current_magnet_state = 0.0  # track magnet state locally
        #-------------------------------------------------
        self.piper = C_PiperInterface(self.can_interface)
        
        

        # Input Queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
            'magnet_on': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )


        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualMagnetState',  # Include the magnet state key here
            ]

        # build ring buffer
        # Ring Buffer for robot state
        example = {
            'ActualTCPPose': np.zeros((6,), dtype=np.float64),
            'robot_receive_timestamp': time.time(),
            'ActualMagnetState': np.zeros((1,), dtype=np.float64),
        }

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.3,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        

    # ========= Launch Methods ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PiperController] Controller process spawned at PID {self.pid}")

    def stop(self, wait=True):
        self.input_queue.put({'cmd': Command.STOP.value})
        if wait:
            self.stop_wait()

    def start_wait(self):
        got_event = self.ready_event.wait(self.launch_timeout)
        if not got_event:
            raise RuntimeError("Piper controller never became ready!")
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    def setup_bluetooth(self):
        try:
            self.bt_connection = serial.Serial(self.bt_port, self.bt_baud_rate)
            time.sleep(2.0)  # Give the connection a couple seconds to initialize
            print("[Electromagnet] Bluetooth connection established.")
        except serial.SerialException as e:
            self.bt_connection = None
            print(f"[Electromagnet] Failed to connect to Bluetooth device: {e}")

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= Context Manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= Command Methods ===========

    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        self.input_queue.put({
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        })
    # [新增] move_point: 新增的高层API，直接put一个 MOVE_POINT 命令
    def move_point(self, pose):
        """Directly move to a specified pose without interpolation."""
        assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (6,)

        self.input_queue.put({
            'cmd': Command.MOVE_POINT.value,
            'target_pose': pose
        })

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def _wait_until_enabled(self, timeout=5):
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            enable_flag = (
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and
                self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and
                self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and
                self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and
                self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and
                self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            if enable_flag:
                print("[PiperController] All motors enabled.")
                break
            if elapsed > timeout:
                print("[PiperController] Enable timeout. Exiting.")
                exit(0)
            time.sleep(0.5)

    def control_esp32(self, magnet_on):
        """
        If your 'right_trigger' or some button is pressed, we interpret that 
        as 'turn electromagnet ON'. If not pressed, turn it OFF.
        
        gripper_value: 0.0 ～1.0
        """
        if not self.bt_connection or not self.bt_connection.is_open:
            print("[Electromagnet] Bluetooth not available or not open.")
            return

        try:
            if int(magnet_on) == 1:
                # print("[Electromagnet] Turning electromagnet ON.")
                self.bt_connection.write(b'1')
                self.current_magnet_state = 1.0
            else:
                self.bt_connection.write(b'0')
                self.current_magnet_state = 0.0
        except serial.SerialException as e:
            print(f"[Electromagnet] Error writing to Bluetooth device: {e}")

    # ========= Main Loop ===========
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
        # init
        self.piper.ConnectPort()
        self.piper.EnableArm(7)
        self._wait_until_enabled()
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self.setup_bluetooth()
        try:
            if self.verbose:
                print("[PiperController] Starting main loop.")

            if self.tcp_offset_pose is not None:
                pass
            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    pass
                else:
                    pass
            # init joints
            if self.joints_init is not None:
                pass #TODO add init joints
            if self.reset:
                self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00) #角度控制模式
                # position = [
                #     -0.0296,
                #     0.7135,
                #     -1.0597,
                #     -0.0159,
                #     1.2376,
                #     1.0212,
                #     0.01
                #     ]
                position = [-0.012, 0.607, -0.587, 0.091, 0.376, 0.524, 0.01]

                factor = 57324.840764 #1000*180/3.14
                joint_0 = round(position[0]*factor)
                joint_1 = round(position[1]*factor)
                joint_2 = round(position[2]*factor)
                joint_3 = round(position[3]*factor)
                joint_4 = round(position[4]*factor)
                joint_5 = round(position[5]*factor)
                self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                print("[PiperController] Reset the robot.")
                time.sleep(2)
                
            # main loop
            dt = 1. / self.frequency
            curr_pose_raw = self.piper.GetArmEndPoseMsgs()

            x = curr_pose_raw.end_pose.X_axis / 1000000
            y = curr_pose_raw.end_pose.Y_axis / 1000000
            z = curr_pose_raw.end_pose.Z_axis / 1000000

            # Convert the SDK’s Euler angles to radians.
            # Note: The SDK outputs angles in 0.001 degrees, so divide by 1000 and convert to radians.
            roll = np.deg2rad(curr_pose_raw.end_pose.RX_axis / 1000)
            pitch = np.deg2rad(curr_pose_raw.end_pose.RY_axis / 1000)
            yaw = np.deg2rad(curr_pose_raw.end_pose.RZ_axis / 1000)
            rotation_vector = R.from_euler('xyz', [roll, pitch, yaw]).as_rotvec()
            curr_pose = [x, y, z, rotation_vector[0], rotation_vector[1], rotation_vector[2]]
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([curr_pose])
            )
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)
                curr_pose_raw = self.piper.GetArmEndPoseMsgs()

                joint = self.piper.GetArmJointMsgs()
                # Convert rotation vector to Euler angles
                rotation_vector = pose_command[3:6]
                euler_angles = R.from_rotvec(rotation_vector).as_euler('xyz', degrees=True)
                magnet_state = self.current_magnet_state
                magnet_state = np.array([magnet_state], dtype=np.float64)


                new_pose = [
                    pose_command[0],
                    pose_command[1],
                    pose_command[2],
                    euler_angles[0],
                    euler_angles[1],
                    euler_angles[2]
                ]
                scaled_pose = [
                    int(new_pose[0] * 1000000),
                    int(new_pose[1] * 1000000),
                    int(new_pose[2] * 1000000),
                    int(new_pose[3] * 1000),
                    int(new_pose[4] * 1000),
                    int(new_pose[5] * 1000)
                ]
                # xyz的单位是mm，rpy的单位是度
                self.piper.MotionCtrl_2(0x01, 0x00, 100,0x00)
                self.piper.EndPoseCtrl(*scaled_pose)
                # Maintain control loop frequency
                # time.sleep(max(0, (1 / self.frequency) - (time.monotonic() - t_now)))
                # update robot state
                state = {
                    'ActualTCPPose': new_pose,
                    'robot_receive_timestamp': time.time(),
                    'ActualMagnetState': magnet_state  # Add magnet state to the ring buffer
                }
                self.ring_buffer.put(state)
                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(f"[PiperController] Scheduled waypoint to {target_pose} at {t_insert}")
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time                        
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.MOVE_POINT.value:
                        # 直接用EndPoseCtrl走点，不做插值
                        target_pose = command['target_pose']
                        pose_interp = pose_interp.move_point(target_pose)
                    elif cmd == Command.MAGNET.value:
                        magnet_on = command['magnet_on']
                        self.control_esp32(magnet_on)
                    else:
                        keep_running = False
                        break

                if iter_idx == 0:
                    self.ready_event.set()

                iter_idx += 1

                if self.verbose:
                    pass
        finally:
            # self.piper.DisableArm(7)
            # self.piper.GripperCtrl(0,1000,0x02, 0)
            self.ready_event.set()
            if self.verbose:
                print("[PiperController] Main loop exited.")

        
