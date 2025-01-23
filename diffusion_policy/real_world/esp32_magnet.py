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