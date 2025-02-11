#!/usr/bin/env python3
import time
from diffusion_policy.encapsulated_oculusReader.oculus_reader import OculusReader


def main():
    # If you're using a network device, set ip_address and port accordingly.
    # For USB-connected devices, you can leave ip_address as None.
    oculus = OculusReader(ip_address=None, port=5555, print_FPS=True, run=True)

    try:
        while True:
            # Get the latest transformations and button states.
            transforms, buttons = oculus.get_transformations_and_buttons()
            
            if buttons:
                right_js = buttons.get('rightJS')
                if right_js is not None:
                    print("Right Joystick:", right_js)
            else:
                print("Waiting for joystick/button data...")

            time.sleep(0.1)  # Adjust the polling rate as needed

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        oculus.stop()

if __name__ == '__main__':
    main()
