import serial
import time

class BluetoothMagnetController:
    def __init__(self, bt_port='/dev/rfcomm0', baud_rate=115200):
        """
        bt_port: Your Bluetooth serial port, e.g. '/dev/rfcomm0' (Linux/RPi).
        baud_rate: Typical baud rate for your ESP32 UART-BT bridge.
        """
        self.bt_port = bt_port
        self.bt_baud_rate = baud_rate
        self.connection = None
        self.electromagnet_state = False  # Tracks ON/OFF state of electromagnet
        self.setup_bluetooth()

    def setup_bluetooth(self):
        try:
            self.connection = serial.Serial(self.bt_port, self.bt_baud_rate)
            time.sleep(2.0)  # Give the connection a couple seconds to initialize
            print("[Electromagnet] Bluetooth connection established.")
        except serial.SerialException as e:
            self.connection = None
            print(f"[Electromagnet] Failed to connect to Bluetooth device: {e}")

    def control_esp32(self, gripper_value):
        """
        If your 'right_trigger' or some button is pressed, we interpret that 
        as 'turn electromagnet ON'. If not pressed, turn it OFF.
        
        gripper_value: 0 or 1 (0=OFF, 1=ON)
        """
        if not self.connection or not self.connection.is_open:
            print("[Electromagnet] Bluetooth not available or not open.")
            return

        try:
            # Turn ON if not ON already
            if gripper_value == 1 and (not self.electromagnet_state):
                self.connection.write(b'1')  # or whatever your ESP32 code expects
                self.electromagnet_state = True
                print("[Electromagnet] Sent ON command to ESP32.")
            # Turn OFF if currently ON
            elif gripper_value != 1 and self.electromagnet_state:
                self.connection.write(b'0')  # or whatever your ESP32 code expects
                self.electromagnet_state = False
                print("[Electromagnet] Sent OFF command to ESP32.")
        except serial.SerialException as e:
            print(f"[Electromagnet] Error sending data to ESP32: {e}")
