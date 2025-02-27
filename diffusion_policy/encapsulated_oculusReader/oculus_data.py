import numpy as np
from collections import deque
from scipy.ndimage import uniform_filter1d
from .oculus_reader.reader import OculusReader  # 假设oculus_reader库已经正确安装
import threading
import time
from rich.live import Live
from rich.table import Table
from rich.console import Console
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

class OculusHandler:
    def __init__(self, oculus_reader, right_controller=True, hz=60, use_filter=True, max_threshold=1.0, orientation_threshold=0.01):
        self.oculus_reader = oculus_reader
        self.hz = hz
        self.use_filter = use_filter
        self.max_threshold = max_threshold
        self.raw_data = deque(maxlen=10)
        self.filtered_data = deque(maxlen=10)

        self.right_controller = right_controller
        self.controller_id = "r" if self.right_controller else "l"

        # Changed orientation increment to 3-element vector
        self.increments = {
            'position': np.zeros(3),
            'orientation': np.zeros(3)
        }
        self.current_orientation = np.zeros((3, 3))

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    #---------------------------------
    @staticmethod
    def rot_mat(angles, hom: bool = False):
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
    @staticmethod
    def mat2quat(rmat):
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
            ]
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
    @staticmethod
    def quat_multiply(quaternion1, quaternion0):
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
            )
        )
    @staticmethod
    def quat_inverse(quaternion):
        conjugate = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
        return conjugate / np.dot(quaternion, quaternion)



    #---------------------------------


    def update(self):
        poses, buttons = self.oculus_reader.get_transformations_and_buttons()
        if not poses or self.controller_id not in poses:
            return

        pose_matrix = poses[self.controller_id]
        transformation_matrix = np.array([
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        pose_matrix = transformation_matrix @ pose_matrix

        position = pose_matrix[:3, 3]
        orientation_matrix = pose_matrix[:3, :3]

        self.raw_data.append((position, orientation_matrix))

        if len(self.raw_data) >= 3 and self.use_filter:
            smoothed_positions = uniform_filter1d(
                np.array([pos for pos, _ in self.raw_data]), size=3, axis=0)
            smoothed_orientations = uniform_filter1d(
                np.array([ori for _, ori in self.raw_data]), size=3, axis=0)
            self.filtered_data.append((smoothed_positions[-1], smoothed_orientations[-1]))
        else:
            self.filtered_data.append((position, orientation_matrix))

        if len(self.filtered_data) >= 2:
            self.increments['position'] = self.filtered_data[-1][0] - self.filtered_data[-2][0]

            prev_ori = self.filtered_data[-2][1]
            current_ori = self.filtered_data[-1][1]

            # Coordinate adjustment from get_action
            rot_adj = self.rot_mat([-np.pi/2, 0, np.pi/2])
            adjusted_prev = rot_adj @ prev_ori
            adjusted_current = rot_adj @ current_ori

            prev_quat = self.mat2quat(adjusted_prev)
            current_quat = self.mat2quat(adjusted_current)

            delta_quat = self.quat_multiply(current_quat, self.quat_inverse(prev_quat))
            delta_quat = delta_quat[[1, 0, 2, 3]]
            delta_quat[0] *= -1
            if delta_quat[3] < 0:
                delta_quat *= -1

            # Convert to rotation vector
            r = R.from_quat(delta_quat)
            rot_vec = r.as_rotvec()

            self.increments['position'] = np.clip(
                self.increments['position'] / self.max_threshold,
                -1.0, 1.0
            )
            self.increments['orientation'] = np.clip(
                rot_vec / self.max_threshold,
                -1.0, 1.0
            )
            self.current_orientation = current_ori
    def get_increment(self):
        return self.increments
    
    def get_original_orientation(self):
        if len(self.raw_data) > 0:
            # raw_data 中存储的应该是 4x4 的 pose 矩阵
            position, pose_matrix = self.raw_data[-1]
            
            # 提取出 3x3 的旋转矩阵 (前三行前三列)
            orientation_matrix = pose_matrix[:3, :3]
            return orientation_matrix  # 返回 3x3 旋转矩阵
        else:
            return None  # 如果没有数据可返回，则返回 None



    def get_current_orientation(self):
        return self.current_orientation

    def get_raw_orientation(self):
        if len(self.raw_data) > 0:
            return self.raw_data[-1][1]
        return np.zeros((3, 3))

    def get_buttons(self):
        _, buttons = self.oculus_reader.get_transformations_and_buttons()
        return buttons

    def run(self):
        interval = 1.0 / self.hz
        while True:
            self.update()
            time.sleep(interval)

def generate_table(target_pos,increment, buttons,speed,pwm_value, current_orientation, raw_orientation):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter")
    table.add_column("Value")
    if 'position' in increment and isinstance(increment['position'], np.ndarray) and increment['position'].shape == (3,):
        table.add_row("Position Increment pos x", f"{increment['position'][0]:.5f}")
        table.add_row("Position Increment pos y", f"{increment['position'][1]:.5f}")
        table.add_row("Position Increment pos z", f"{increment['position'][2]:.5f}")
    else:
        table.add_row("Position Increment", "Invalid data")

    # 添加目标位置
    if isinstance(target_pos, np.ndarray) and target_pos.shape == (3,):
        table.add_row("Target pos x", f"{target_pos[0]:.5f}")
        table.add_row("Target pos y", f"{target_pos[1]:.5f}")
        table.add_row("Target pos z", f"{target_pos[2]:.5f}")
    else:
        table.add_row("Target pos", "Invalid data")


    table.add_row("speed:", str(speed))
    table.add_row("pwm:", str(pwm_value))

    # 添加方向增量
    if 'orientation' in increment and isinstance(increment['orientation'], np.ndarray) and increment['orientation'].shape == (3, 3):
        table.add_row("Orientation Increment row 1", str(increment['orientation'][0]))
        table.add_row("Orientation Increment row 2", str(increment['orientation'][1]))
        table.add_row("Orientation Increment row 3", str(increment['orientation'][2]))
    else:
        table.add_row("Orientation Increment", "Invalid data")

    # 添加当前方向
    if isinstance(current_orientation, np.ndarray) and current_orientation.shape == (3, 3):
        table.add_row("Current Orientation row 1", str(current_orientation[0]))
        table.add_row("Current Orientation row 2", str(current_orientation[1]))
        table.add_row("Current Orientation row 3", str(current_orientation[2]))
    else:
        table.add_row("Current Orientation", "Invalid data")

    # 添加原始方向
    if isinstance(raw_orientation, np.ndarray) and raw_orientation.shape == (3, 3):
        table.add_row("Raw Orientation row 1", str(raw_orientation[0]))
        table.add_row("Raw Orientation row 2", str(raw_orientation[1]))
        table.add_row("Raw Orientation row 3", str(raw_orientation[2]))
    else:
        table.add_row("Raw Orientation", "Invalid data")
    

    table.add_row("Buttons", str(buttons))
    return table

# Main function
if __name__ == "__main__":
    hz = 10
    ocu_hz = 20
    oculus_reader = OculusReader()
    handler = OculusHandler(oculus_reader, right_controller=True, hz=ocu_hz, use_filter=False, max_threshold=0.7/ocu_hz)
    console = Console()
    max_speed = 40000
    #client.send_cmd(3)

    time.sleep(1)
    input("press return to conitue")
    
    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                time.sleep(1 / hz)

                # input("press return to conitue")
                

                increment = handler.get_increment()
                buttons = handler.get_buttons()
                current_orientation = handler.get_current_orientation()
                raw_orientation = handler.get_raw_orientation()

                # buttons, position, speed, acc = iter
                right_trigger = buttons.get("rightTrig", [0])[0]
                pwm_value = 2500 - right_trigger * 2000
                # 计算归一化因子
                normalization_factor = np.sqrt(3)

                # 归一化位置分量
                normalized_increment = np.linalg.norm(increment['position'])/ normalization_factor
                
                # if buttons['A'] == True:
                #     increment['position'] = np.zeros(3)


                # target_pos[:3] += 0.66*increment['position']*((max_speed/60)/1000)/hz

                robot_speed = normalized_increment*(max_speed)
                if robot_speed<1000:
                    robot_speed = 1000
                robot_speed = math.ceil(robot_speed) 
                
                # print("tsrget pos",target_pos)
                print("increment:", increment['position']*((max_speed/60)/1000)/hz)
                # # Update progress bar
                # progress.update(task1, completed=min(robot_speed / 100, 100))
                # progress.update(task2, completed=min(robot_acc / 100, 100))
                # Generate and update the table
                # client.linear_goto_pos_2d_mix_vr(target_pos[0], target_pos[1],target_pos[2],2500 - buttons["rightTrig"][0]*2000,None,robot_speed)



                # table = generate_table(target_pos,increment, buttons, robot_speed,pwm_value, current_orientation, raw_orientation)
                # live.update(table, refresh=True)

    except KeyboardInterrupt:
        print("Terminated by user.")
