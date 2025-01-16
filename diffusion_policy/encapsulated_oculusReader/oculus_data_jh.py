try:
    from oculus_reader.reader import OculusReader
except ImportError:
    raise Exception(
        "Please install oculus_reader following https://github.com/rail-berkeley/oculus_reader"
    )

import numpy as np

import logging
import time


class OculusInterface:
    """Define Oculus interface to control Franka Panda.

    To control the robot:
    - The button on the side (thumb) can toggle gripper state (open and close).
    - The button B will reset the environment and robot.
    - Movement is continuously tracked without needing to press the trigger button.
    """

    def __init__(self, oculus_reader,realtive_move=True):
        self.oculus = oculus_reader
        # Allow some time for OculusReader to initialize and gather data
        time.sleep(1)  # Adjust as necessary based on OculusReader's behavior
        self.reset()
        self.positional_move_only = False
        self.relative_move = realtive_move

    def reset(self):
        self.prev_oculus_pos = None
        self.prev_oculus_quat = None

    def _get_action(self, oculus_pos, oculus_quat):
        if self.prev_oculus_pos is None or self.prev_oculus_quat is None:
            return np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        rel_oculus_pos = self.prev_oculus_pos - oculus_pos

        # Relative movement speed between VR and robot
        action_pos = rel_oculus_pos

        # Swap and flip axes
        action_pos = action_pos[[2, 0, 1]]
        action_pos[2] = -action_pos[2]
        action_pos *= 1  # Scale to make the delta noticeable

        rel_oculus_quat = self.quat_multiply(
            oculus_quat, self.quat_inverse(self.prev_oculus_quat)
        )

        action_quat = np.array(list(rel_oculus_quat), dtype=np.float32)  # xyzw

        action_pos = np.clip(
            action_pos, -0.1, 0.1
        )  # Maximum movement is 0.1m per step

        if action_quat[3] < 0.0:
            action_quat *= -1.0

        return action_pos, action_quat

    def get_action(self, use_quat=True):
        assert use_quat, "Oculus only works with quaternion"

        poses, buttons = self.oculus.get_transformations_and_buttons()

        button_A = buttons.get("A", False)
        button_B = buttons.get("B", False)
        gripper_press = buttons.get("RTr", False) # 手柄后面的按钮
        handle_press = buttons.get("RG", False) # 大拇指处的按钮
        joystick_press = buttons.get("RJ", False) # 按下遥感
        action_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        action_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw
        

        # Access the 'r' key to get the pose array
        oculus_pose_array = poses.get('r', None)
        if oculus_pose_array is None:
            return np.zeros(7, dtype=np.float32), button_A, button_B, gripper_press # Adjust the size if necessary

        # Always process oculus_pose regardless of handle_press
            
        oculus_pos = oculus_pose_array[:3, 3]
        # Swap and flip axes
        oculus_mat = self.rot_mat([-np.pi / 2, 0, np.pi / 2]) @ oculus_pose_array[:3, :3]
        oculus_quat = self.mat2quat(oculus_mat)

        # Compute action based on current and previous pose
        if self.relative_move:
            action_pos, action_rot = self._get_action(oculus_pos, oculus_quat)
        else:
            action_pos = oculus_pos
            action_rot = oculus_quat
        # Update previous pose for next action computation
        self.prev_oculus_pos = oculus_pos
        self.prev_oculus_quat = oculus_quat

        if self.positional_move_only:
            action_rot = np.array([0, 0, 0, 1], dtype=np.float32)

        action = np.hstack([action_pos, action_rot])

        return action, button_A, button_B, gripper_press

    def print_usage(self):
        print("==============Oculus Usage=================")
        print("Movement: Controlled continuously without pressing the trigger button")
        print("Button on the back: Press to toggle gripper state (open/close)")

    def close(self):

        self.oculus.stop()


    ####################################################################################################
    # Transform functions
    def quat_multiply(self, quaternion1, quaternion0):
        """Return multiplication of two quaternions.
        >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
        >>> np.allclose(q, [-44, -14, 48, 28])
        True
        """
        x0, y0, z0, w0 = quaternion0
        x1, y1, z1, w1 = quaternion1
        return np.array(
            (
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            ),
            dtype=np.float32,
        )

    def quat_conjugate(self, quaternion):
        """Return conjugate of quaternion.
        >>> q0 = random_quaternion()
        >>> q1 = quat_conjugate(q0)
        >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
        True
        """
        return np.array(
            (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
            dtype=np.float32,
        )

    def quat_inverse(self, quaternion):
        """Return inverse of quaternion.
        >>> q0 = random_quaternion()
        >>> q1 = quat_inverse(q0)
        >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
        True
        """
        return self.quat_conjugate(quaternion) / np.dot(quaternion, quaternion)

    def rot_mat(self, angles, hom: bool = False):
        """Given @angles (x, y, z), compute rotation matrix
        Args:
            angles: (x, y, z) rotation angles.
            hom: whether to return a homogeneous matrix.
        """
        x, y, z = angles
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(x), -np.sin(x)],
                       [0, np.sin(x), np.cos(x)]], dtype=np.float32)
        Ry = np.array([[np.cos(y), 0, np.sin(y)],
                       [0, 1, 0],
                       [-np.sin(y), 0, np.cos(y)]], dtype=np.float32)
        Rz = np.array([[np.cos(z), -np.sin(z), 0],
                       [np.sin(z), np.cos(z), 0],
                       [0, 0, 1]], dtype=np.float32)

        R = Rz @ Ry @ Rx
        if hom:
            M = np.zeros((4, 4), dtype=np.float32)
            M[:3, :3] = R
            M[3, 3] = 1.0
            return M
        return R

    def mat2quat(self, rmat):
        """
        Converts given rotation matrix to quaternion.
        Args:
            rmat: 3x3 rotation matrix
        Returns:
            vec4 float quaternion angles
        """
        M = np.asarray(rmat).astype(np.float32)[:3, :3]

        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ],
            dtype=np.float32,
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
        if q1[0] < 0.0:
            np.negative(q1, q1)
        inds = np.array([1, 2, 3, 0])
        return q1[inds]


def main():
    oculus_reader = OculusReader()
    oculus = OculusInterface(oculus_reader)
    oculus.print_usage()
    try:
        while True:
            action, button_A, button_B,gripper = oculus.get_action()
            print("Action:", action)
            if button_A:
                print("Button A pressed.")
            if button_B:
                print("Button B pressed.")
            if gripper:
                print("Gripper closed.")
            time.sleep(0.1)  # Add a small delay to prevent overwhelming the output
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        oculus.close()


if __name__ == "__main__":
    main()
