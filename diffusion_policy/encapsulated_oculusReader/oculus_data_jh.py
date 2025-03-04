#!/usr/bin/env python3
"""
OculusInterface for non-ROS robot control that provides incremental changes only.

This code continuously collects data from an OculusReader into thread-safe buffers.
Each call to get_action() computes the incremental change in pose between the two latest samples,
returning a 7D vector:
  [dx, dy, dz, qx, qy, qz, qw]
where:
  - dx, dy, dz: incremental position change in meters.
  - qx, qy, qz, qw: incremental rotation as a quaternion.
    (You can convert the quaternion to Euler angles (roll, pitch, yaw in degrees) using quat_to_euler().)

It also returns the latest button state (a dictionary) aligned in time with the pose.
"""

import time
import numpy as np
import math
import threading
from collections import deque

try:
    from .oculus_reader.reader import OculusReader
except ImportError:
    raise Exception("Please install oculus_reader following https://github.com/rail-berkeley/oculus_reader")
from scipy.spatial.transform import Rotation
class OculusInterface:
    def __init__(self, oculus_reader, max_buffer=10, relative_move=True,hz=100):
        """
        Initialize the OculusInterface.

        Args:
            oculus_reader: An instance of OculusReader.
            max_buffer: Maximum number of samples to keep in the buffers.
            relative_move: (Always True here) Only incremental changes are computed.
        """
        self.hz = hz    
        self.oculus = oculus_reader
        # Allow some time for OculusReader to initialize.
        time.sleep(1)
        self.relative_move = relative_move
        # Thread-safe data buffers.
        self.pose_buffer = deque(maxlen=max_buffer)
        self.button_buffer = deque(maxlen=max_buffer)
        self.lock = threading.Lock()
        # Start data collection thread.
        self.running = True
        self.read_thread = threading.Thread(target=self._read_data)
        self.read_thread.daemon = True
        self.read_thread.start()

    def _read_data(self):
        """
        Continuously read data from OculusReader and store it in buffers.
        Instead of storing the full 4x4 homogeneous matrix, this function extracts
        the translation (xyz) and converts the rotation matrix to a quaternion (qx, qy, qz, qw),
        and stores the combined 7-element vector [x, y, z, qx, qy, qz, qw] in pose_buffer.
        We only use the 'r' (right-hand) controller's pose.
        """
        while self.running:
            poses, buttons = self.oculus.get_transformations_and_buttons()
            if not poses:
                print("No poses received.")
                time.sleep(0.1)
                continue
            # Get the right-hand controller's pose (4x4 matrix)
            pose = poses.get("r")
            if pose is not None:
                # Extract translation (position)
                pos = pose[:3, 3].copy()
                pos = pos[[2, 0, 1]]
                pos[0] = -pos[0]
                pos[1] = -pos[1]
                # Extract rotation matrix (upper-left 3x3 block)
                oculus_mat = self.rot_mat([-np.pi / 2, 0, np.pi / 2]) @ pose[:3, :3]
                # Convert rotation matrix to quaternion (using a helper function, e.g., self.mat2quat)
                quat = self.mat2quat(oculus_mat)
                quat = quat[[0, 1, 2, 3]]
                if quat[3] < 0.0:
                    quat *= -1.0
                # Combine position and quaternion into a 7-element vector.
                pose_7d = np.hstack([pos, quat])
                with self.lock:
                    self.pose_buffer.append(pose_7d)
                    # Make a shallow copy of the buttons dictionary.
                    self.button_buffer.append(buttons.copy() if buttons is not None else {})
            else:
                print("Pose for 'r' not found; skipping this sample.")
            time.sleep(1.0 / self.hz)

    def get_action(self):
        """
        Get the latest incremental pose change and button state.

        Returns:
            action: A numpy array with 7 elements [dx, dy, dz, qx, qy, qz, qw],
                    where the position change is in meters and the rotation is given as a quaternion.
            buttons: A dictionary of button states.

        If fewer than two samples exist, it uses the last available sample for both previous and current,
        which results in a zero translation and an identity quaternion (0,0,0,1).
        """
        with self.lock:
            if len(self.pose_buffer) == 0:
                # No pose data available at all.
                buttons = {}
                return np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), buttons
            elif len(self.pose_buffer) < 2:
                # Only one sample exists; use it for both previous and current.
                pose_prev = self.pose_buffer[-1].copy()
                pose_curr = self.pose_buffer[-1]
                buttons = self.button_buffer[-1].copy() if len(self.button_buffer) > 0 else {}
            else:
                # Use the two most recent samples.
                pose_prev = self.pose_buffer[-2]
                pose_curr = self.pose_buffer[-1]
                buttons = self.button_buffer[-1].copy()

        # Compute incremental position: difference in the translation part.
        pos_prev = pose_prev[:3]
        pos_curr = pose_curr[:3]
        delta_pos = pos_curr - pos_prev
        delta_pos *= 1.4
        # Limit the maximum position change (e.g., 0.1 m per update).
        delta_pos = np.clip(delta_pos, -0.1, 0.1)
        

        # Compute incremental rotation.
        quat_prev = pose_prev[3:]
        quat_curr = pose_curr[3:]
        # Compute relative rotation as: delta_quat = quat_curr * inverse(quat_prev)
        delta_quat = self.quat_multiply(quat_curr, self.quat_inverse(quat_prev))
        
        # # change some axis
        # delta_quat = delta_quat[[1, 0, 2 , 3]]
        # delta_quat[0] = -delta_quat[0]
        # delta_quat = np.array(list(delta_quat))
        
        # if delta_quat[3] < 0.0:
        #     delta_quat *= -1.0


        # Combine position and rotation increments.
        action = np.hstack([delta_pos, delta_quat])
        return action.astype(np.float32), buttons

    def close(self):
        """Stop the data collection thread and close the OculusReader."""
        self.running = False
        self.read_thread.join(timeout=1)
        self.oculus.stop()

    # --- Helper functions for quaternion and matrix operations ---
    def rot_mat(self,angles, hom: bool = False):
        """Given @angles (x, y, z), compute rotation matrix
        Args:
            angles: (x, y, z) rotation angles.
            hom: whether to return a homogeneous matrix.
        """
        x, y, z = angles
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

        R = Rz @ Ry @ Rx
        if hom:
            M = np.zeros((4, 4), dtype=np.float32)
            M[:3, :3] = R
            M[3, 3] = 1.0
            return M
        return R
    def quat_multiply(self, q1, q0):
        """
        Multiply two quaternions (order: q1 * q0) where each is (x, y, z, w).
        """
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1
        return np.array([
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
        ], dtype=np.float32)

    def quat_conjugate(self, q):
        """Return the conjugate of quaternion q (x, y, z, w)."""
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    def quat_inverse(self, q):
        """Return the inverse of quaternion q (x, y, z, w)."""
        return self.quat_conjugate(q) / np.dot(q, q)

    def mat2quat(self, rmat):
        """
        Convert a 3x3 rotation matrix to a quaternion (x, y, z, w).

        This follows a method similar to the reference implementation.
        """
        M = np.asarray(rmat, dtype=np.float32)[:3, :3]
        m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
        m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
        m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
        # Build symmetric K matrix.
        K = np.array([
            [m00 - m11 - m22, 0.0,          0.0,          0.0],
            [m01 + m10,       m11 - m00 - m22, 0.0,          0.0],
            [m02 + m20,       m12 + m21,       m22 - m00 - m11, 0.0],
            [m21 - m12,       m02 - m20,       m10 - m01,       m00 + m11 + m22]
        ], dtype=np.float32)
        K /= 3.0
        # Eigen-decomposition; the quaternion is the eigenvector corresponding to the largest eigenvalue.
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q = V[inds, np.argmax(w)]
        if q[0] < 0.0:
            q = -q
        inds = np.array([1, 2, 3, 0])
        return q[inds]

    def quat_to_euler(self, q):
        """
        Convert a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in degrees.
        Uses the standard Tait-Bryan angles (x-y-z convention).
        """
        x, y, z, w = q
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)], dtype=np.float32)

# --- Test code mapping the increments to a robot target state ---
def main():
    """
    Test routine:
      - Creates an OculusReader and wraps it in OculusInterface.
      - In a loop (10 Hz), obtains the incremental pose change and button info.
      - Maps the increments onto a simulated robot target state (position in meters and rotation in Euler angles in degrees).
    """
    oculus_reader = OculusReader()
    oculus_interface = OculusInterface(oculus_reader)
    
    # Initialize the robot's target state.
    robot_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    robot_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll, pitch, yaw in degrees
    
    print("Starting OculusInterface test loop (Press Ctrl+C to exit)...")
    
    try:
        while True:
            # Get the incremental action and button state.
            # action is [dx, dy, dz, qx, qy, qz, qw]
            action, buttons = oculus_interface.get_action()
            delta_pos = action[:3]
            delta_quat = action[3:]
            
            # Convert robot euler(degrees) to quaternion
            robot_quart = Rotation.from_euler('xyz', robot_euler, degrees=True).as_quat()
            
            # Map the increments onto the robot's target state.
            robot_target_position = robot_position + delta_pos
            robot_target_quart = oculus_interface.quat_multiply(robot_quart,delta_quat)
            robot_targrt_euler = Rotation.from_quat(robot_target_quart).as_euler('xyz', degrees=True)
            print("Increment:")
            print("  Position change (meters):", delta_pos)
            print("  Rotation change (degrees):", delta_quat)
            print("Updated Robot Target State:")
            print("  Position (meters):", robot_target_position)
            print("  Rotation (degrees):", robot_targrt_euler)
            print("Button states:", buttons)
            print("-----------------------------------------------------")
            
            time.sleep(0.1)  # Loop at approximately 10 Hz.
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        oculus_interface.close()

if __name__ == "__main__":
    main()
