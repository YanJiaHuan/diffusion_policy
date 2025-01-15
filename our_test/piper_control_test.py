import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from diffusion_policy.real_world.piper_controller import PiperInterpolationController  # Adjust if needed

def main():
    # Start the shared memory manager
    with SharedMemoryManager() as shm_manager:
        # Initialize the Piper Controller
        controller = PiperInterpolationController(
            shm_manager=shm_manager,
            can_interface="can_piper",   # Adjust if needed
            frequency=100,               # Control frequency in Hz
            max_pos_speed=0.25,          # Max linear speed (m/s)
            max_rot_speed=0.16,          # Max rotational speed (rad/s)
            verbose=True                 # Enable detailed logging
        )

        # Start the controller
        controller.start(wait=True)

        # Wait for the controller to initialize
        if controller.is_ready:
            print("[Test] Controller is ready.")
        else:
            print("[Test] Controller failed to start.")
            return

        try:
            # === 1. Get the current robot state ===
            state = controller.get_state()
            if state is None:
                print("[Test] Failed to retrieve robot state.")
                return

            actual_pose = state['ActualTCPPose']
            print(f"[Test] Current TCP Pose: {actual_pose}")

            # === 2. Move the robot to a target pose ===
            # Example target pose: [X, Y, Z, RX, RY, RZ]
            # Units: meters for XYZ, degrees for RPY (converted to radians inside the controller)
            target_pose = [0.05, 0.0, 0.206, 0.0, 85, 0.0]  # Move up in Z and rotate around Y-axis
            scheduled_time = time.time() + 2  # Execute after 2 seconds

            print(f"[Test] Scheduling waypoint to: {target_pose}")
            controller.schedule_waypoint(target_pose, scheduled_time)

            # Wait for the movement to complete
            time.sleep(5)  # Wait extra to ensure motion finishes

            # === 3. Check the updated pose ===
            updated_state = controller.get_state()
            if updated_state is None:
                print("[Test] Failed to retrieve updated robot state.")
                return

            updated_pose = updated_state['ActualTCPPose']
            print(f"[Test] New TCP Pose: {updated_pose}")

            # === 4. Idle to keep the robot active ===
            print("[Test] Entering idle state. Press Ctrl+C to stop.")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("[Test] Interrupted by user.")
        except Exception as e:
            print(f"[Test] Error occurred: {e}")
        finally:
            pass

if __name__ == "__main__":
    main()
