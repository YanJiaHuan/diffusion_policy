import serial
import time
import multiprocessing as mp
from diffusion_policy.shared_memory.shared_memory_queue import Empty

class BluetoothMagnetController:
    def __init__(self, bt_port='/dev/rfcomm0', baud_rate=115200):
        """
        bt_port: Your Bluetooth serial port, e.g. '/dev/rfcomm0' (Linux/RPi).
        baud_rate: Typical baud rate for your ESP32 UART-BT bridge.
        """
        self.bt_port = bt_port
        self.bt_baud_rate = baud_rate
        self.connection = None
        self.electromagnet_state = False 
        self.setup_bluetooth()

    def setup_bluetooth(self):
        try:
            self.connection = serial.Serial(self.bt_port, self.bt_baud_rate)
            time.sleep(2.0)  # Give the connection a couple seconds to initialize
            print("[Electromagnet] Bluetooth connection established.")
        except serial.SerialException as e:
            self.connection = None
            print(f"[Electromagnet] Failed to connect to Bluetooth device: {e}")

    def control_esp32(self, magnet_on):
        """
        If your 'right_trigger' or some button is pressed, we interpret that 
        as 'turn electromagnet ON'. If not pressed, turn it OFF.
        
        gripper_value: 0.0 ～1.0
        """
        if not self.connection or not self.connection.is_open:
            print("[Electromagnet] Bluetooth not available or not open.")
            return

        try:
            if int(magnet_on) == 1:
                self.connection.write(b'1')
                self.electromagnet_state = True
            else:
                self.connection.write(b'0')
                self.electromagnet_state = False
        except serial.SerialException as e:
            print(f"[Electromagnet] Error writing to Bluetooth device: {e}")

    def get_magnet_state(self):
        """
        Instead of returning a boolean, return a float between 0.0 and 1.0,
        where 1.0 represents the magnet being fully "on" and 0.0 represents it being "off".
        This could be determined based on various criteria, such as signal strength, magnet's actual status, etc.
        """
        # For simplicity, assuming 0.0 to 1.0 based on electromagnet state:
        # You can replace this logic with more complex models of how the magnet's state probability is determined.
        if self.electromagnet_state:
            return 1.0
        else:
            return 0.0
        

def magnet_controller_process(command_queue: mp.Queue, state_queue: mp.Queue, bt_port='/dev/rfcomm0', baud_rate=115200):
    """
    独立的磁铁控制进程。
    """
    magnet = BluetoothMagnetController(bt_port=bt_port, baud_rate=baud_rate)
    while True:
        try:
            command = command_queue.get(timeout=0.1)  # 非阻塞等待命令
            if command == 'STOP':
                print("[MagnetController] Stopping magnet controller process.")
                break
            elif isinstance(command, dict):
                action = command.get('action')
                if action == 'control_esp32':
                    magnet_on = command.get('value', 0)
                    magnet.control_esp32(magnet_on)
        except Empty:
            pass  # 没有新命令，继续

        # 定期发送当前磁铁状态
        current_state = magnet.get_magnet_state()
        state_queue.put({'magnet_state': current_state})

    # 清理
    if magnet.connection and magnet.connection.is_open:
        magnet.connection.close()
    print("[MagnetController] Magnet controller process terminated.")
