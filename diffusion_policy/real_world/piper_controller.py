import os
import time
import enum
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from piper_sdk import *
import serial
from scipy.spatial.transform import Rotation as R


# ============================== #
#           COMMANDS             #
# ============================== #
class Command(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1
    MOVE_POINT = 2
    MAGNET = 3

# ============================== #
#        PIPER CONTROLLER        #
# ============================== #
class PiperInterpolationController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 can_interface="can_piper",
                 bt_port='/dev/rfcomm0',
                 bt_baud_rate=115200,
                 frequency=125,
                 launch_timeout=5,
                 joints_init=True,
                 joints_init_speed=1.0,
                 soft_real_time=False,
                 max_pos_speed = 0.25,
                 max_rot_speed = 0.6,
                 verbose=False,
                 receive_keys=None,
                 get_max_k=128):

        # assert 0<frequency<= 500
        super().__init__(name="PiperInterpolationController")
        self.can_interface = can_interface
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.prev_pose = None
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed

        #---------------Magnet Controller----------------
        self.bt_port = bt_port
        self.bt_baud_rate = bt_baud_rate
        self.bt_connection = None  # will hold serial.Serial object
        self.current_magnet_state = 0.0  # track magnet state locally
        #-------------------------------------------------
        self.piper = C_PiperInterface_V2(self.can_interface)
        
        # Input Queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': np.zeros((7,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
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

                'TargetTCPPose',
                'TargetMagnetState',
            ]
        # build ring buffer
        # Ring Buffer for robot state
        example = {
            'ActualTCPPose': np.zeros((7,), dtype=np.float64),
            'robot_receive_timestamp': time.time(),
            'ActualMagnetState': np.zeros((1,), dtype=np.float64),
        }

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
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
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
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
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    # [新增] move_point: 新增的高层API，直接put一个 MOVE_POINT 命令
    def move_point(self, pose):
        """Directly move to a specified pose without interpolation."""
        assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.MOVE_POINT.value,
            'target_pose': pose
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def enable_fun(self, piper:C_PiperInterface_V2):
        '''
        使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("使能状态:",enable_flag)
            piper.EnableArm(7)
            piper.GripperCtrl(0,1000,0x01, 0)
            print("--------------------")
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)
            pass
        
        if(elapsed_time_flag):
            print("程序自动使能超时,退出程序")
            exit(0)

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
        self.enable_fun(piper=self.piper)
        self.piper.GripperCtrl(0,1000,0x01, 0)
        self.setup_bluetooth()
        try:
            if self.verbose:
                print("[PiperController] Starting main loop.")
            # init joints
            if self.joints_init:
                print("[PiperController] Initializing joints.")
                factor = 57324.840764 #1000*180/3.14
                position = [-0.012, 0.607, -0.587, 0.091, 0.376, 0.524, 0.01]
                joint_0 = round(position[0]*factor)
                joint_1 = round(position[1]*factor)
                joint_2 = round(position[2]*factor)
                joint_3 = round(position[3]*factor)
                joint_4 = round(position[4]*factor)
                joint_5 = round(position[5]*factor)
                joint_6 = round(position[6]*1000*1000)
                self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                time.sleep(2)
                print("[PiperController] Joints initialized.")
                
            # main loop
            dt = 1. / self.frequency
            pose = self.piper.GetArmEndPoseMsgs()

            x = pose.end_pose.X_axis / 1000000
            y = pose.end_pose.Y_axis / 1000000
            z = pose.end_pose.Z_axis / 1000000
            # Convert the SDK’s Euler angles to radians.
            # Note: The SDK outputs angles in 0.001 degrees, so divide by 1000 and convert to radians.
            roll = pose.end_pose.RX_axis / 1000
            pitch = pose.end_pose.RY_axis / 1000
            yaw = pose.end_pose.RZ_axis / 1000
            
            rotation_vector = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_rotvec()
            curr_magnet = self.current_magnet_state
            curr_pose = [x, y, z, rotation_vector[0], rotation_vector[1], rotation_vector[2],curr_magnet]
            state = {
                    'ActualTCPPose': curr_pose,
                    'robot_receive_timestamp': time.time(),
                    'ActualMagnetState': curr_magnet  # Add magnet state to the ring buffer
                }
            self.ring_buffer.put(state)
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
                # Convert rotation vector to Euler angles
                rotation_vector = pose_command[3:6]
                euler_angles = R.from_rotvec(rotation_vector).as_euler('xyz', degrees=True)
                target_pose = [
                    round(pose_command[0] * 1000000),
                    round(pose_command[1] * 1000000),
                    round(pose_command[2] * 1000000),
                    round(euler_angles[0] * 1000),
                    round(euler_angles[1] * 1000),
                    round(euler_angles[2] * 1000)
                ]
                # xyz的单位是mm，rpy的单位是度
                self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                self.piper.EndPoseCtrl(*target_pose)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
                magnet_command = pose_command[6]
                self.control_esp32(magnet_command) # Control and Update magnet state

                # update robot state
                state = dict()
                magnet_state = self.current_magnet_state
                magnet_state = np.array([magnet_state], dtype=np.float64)
                pose = self.piper.GetArmEndPoseMsgs()
                x = pose.end_pose.X_axis / 1000000
                y = pose.end_pose.Y_axis / 1000000
                z = pose.end_pose.Z_axis / 1000000
                # Convert the SDK’s Euler angles to radians.
                # Note: The SDK outputs angles in 0.001 degrees, so divide by 1000 and convert to radians.
                roll = pose.end_pose.RX_axis / 1000
                pitch = pose.end_pose.RY_axis / 1000
                yaw = pose.end_pose.RZ_axis / 1000
                rotation_vector = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_rotvec()
                curr_pose = [x, y, z, rotation_vector[0], rotation_vector[1], rotation_vector[2],magnet_state]
                state = {
                    'ActualTCPPose': curr_pose,
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
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time                        
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed = self.max_pos_speed, 
                            max_rot_speed = self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.MOVE_POINT.value:
                        # 直接用EndPoseCtrl走点，不做插值
                        target_pose = command['target_pose']
                        pose_interp = pose_interp.move_point(target_pose)
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_end = time.monotonic()
                elapsed_time = t_end - t_now
                if elapsed_time < dt:
                    time.sleep(dt - elapsed_time)

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

        
