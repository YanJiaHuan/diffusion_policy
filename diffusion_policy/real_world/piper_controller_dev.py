import os
import time
import enum
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from piper_sdk import C_PiperInterface
import serial
from scipy.spatial.transform import Rotation as R

# ============================== #
#           COMMANDS             #
# ============================== #
class Command(enum.Enum):
    STOP = 0
    MOVE_POINT = 1
# ============================== #
#        PIPER CONTROLLER        #
# ============================== #
class PiperController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 can_interface = "can_piper",
                 bt_port = '/dev/rfcomm0',
                 bt_baud_rate=115200,
                 frequency=10,
                 reset = True,
                 soft_real_time=False,
                 verbose=False,
                 receive_keys=None,
                 launch_timeout=3,
                 get_max_k=128):
        # assert 0<frequency<=500
        super().__init__(name="PiperController")
        self.can_interface = can_interface
        self.bt_port = bt_port
        self.bt_baud_rate = bt_baud_rate
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.reset = reset
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.bt_connection = None # will hold serial.Serial object
        self.current_magnet_state = 0.0  # track magnet state locally
        self.piper = C_PiperInterface(can_interface)

        # Input Queue
        example = {
            'cmd': Command.MOVE_POINT.value,
            'target_pose': np.zeros((7,), dtype=np.float64),
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
            time.sleep(1.0)  # Give the connection a couple seconds to initialize
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
    def move_point(self, pose):
        pose = np.array(pose)
        self.input_queue.put({
            'cmd': Command.MOVE_POINT.value,
            'target_pose': pose,
        })

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()
        
    def wait_until_enabled(self, timeout=5):
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
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
        # Initialize the Piper interface
        self.piper.ConnectPort()
        self.piper.EnableArm(7)
        self.wait_until_enabled()
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self.setup_bluetooth()
        try:
            if self.verbose:
                print("[PiperController] Starting main loop.")
            if self.reset:
                self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00) #角度控制模式
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
            
            # Main loop
            dt = 1.0 / self.frequency
            iter_idx = 0
            keep_running = True
            while keep_running:
                # fetch robot state
                robot_pose = self.piper.GetArmEndPoseMsgs()
                x = robot_pose.end_pose.X_axis / 1000000
                y = robot_pose.end_pose.Y_axis / 1000000
                z = robot_pose.end_pose.Z_axis / 1000000
                roll = robot_pose.end_pose.RX_axis / 1000
                pitch = robot_pose.end_pose.RY_axis / 1000
                yaw = robot_pose.end_pose.RZ_axis / 1000
                magnet_state = np.array([self.current_magnet_state], dtype=np.float64)
                state = {
                    'ActualTCPPose': np.array([x, y, z, roll, pitch, yaw], dtype=np.float64),
                    'ActualMagnetState': magnet_state,
                    'robot_receive_timestamp': time.time()
                }
                self.ring_buffer.put(state)

                with open('/home/zcai/jh_workspace/diffusion_policy/robot_state.txt', 'a') as f:
                    f.write(f"{x},{y},{z},{roll},{pitch},{yaw},{self.current_magnet_state}\n")


                # fetch command
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
                    elif cmd == Command.MOVE_POINT.value:
                        target_pose = command['target_pose']
                        # convert the euler into rotation vector back to robot controller
                        magnet_command = target_pose[6]
                        self.control_esp32(magnet_command)
                        scaled_pose = [
                            int(target_pose[0] * 1000000),
                            int(target_pose[1] * 1000000),
                            int(target_pose[2] * 1000000),
                            int(target_pose[3] * 1000),
                            int(target_pose[4] * 1000),
                            int(target_pose[5] * 1000)
                        ]
                        self.piper.MotionCtrl_2(0x01, 0x00, 100,0x00)
                        self.piper.EndPoseCtrl(*scaled_pose)
                    else:
                        keep_running = False
                        break


                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                if self.verbose:
                    # print(f"[PiperController] Robot state: {state}")
                    pass

        finally:
            # self.piper.DisableArm(7)
            # self.piper.GripperCtrl(0,1000,0x02, 0)
            self.ready_event.set()
            if self.verbose:
                print("[PiperController] Main loop exited.")