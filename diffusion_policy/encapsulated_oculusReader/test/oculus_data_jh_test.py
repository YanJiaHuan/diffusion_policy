# filepath: /home/zcai/jh_workspace/diffusion_policy/tests/test_oculus_data_jh.py
import pytest
import numpy as np
from diffusion_policy.encapsulated_oculusReader.oculus_data_jh import OculusInterface


class MockOculusReader:
    """
    模拟 OculusReader 的简单示例：
    get_transformations_and_buttons 返回一个 poses 字典
    （内含名为'r' 和 'l' 的 4x4 矩阵用来表示右手和左手），
    以及一个 buttons 字典。
    """
    def __init__(self):
        pass

    def get_transformations_and_buttons(self):
        # 右手和左手的简单转换矩阵示例（单位矩阵或略作变换）
        pose_r = np.eye(4, dtype=np.float32)
        pose_l = np.eye(4, dtype=np.float32)
        pose_r[0, 3] = 0.1  # 在 x 方向移动 0.1
        pose_r[1, 3] = 0.2  # 在 y 方向移动 0.2
        pose_r[2, 3] = 0.3  # 在 z 方向移动 0.3
        poses = {"r": pose_r, "l": pose_l}
        buttons = {"A": [1], "B": [0]}
        return poses, buttons

    def stop(self):
        pass

@pytest.fixture
def oculus_interface():
    mock_reader = MockOculusReader()
    # 创建接口实例, 默认以 degree=True
    return OculusInterface(mock_reader, degree=True)

def test_get_action(oculus_interface):
    """
    测试 get_action() 方法能否返回 (2,7) 的数组和正确的 button 状态
    """
    action, buttons = oculus_interface.get_action()
    # action 应该为 shape=(2,7)，分别对应右手和左手
    assert action.shape == (2,7)
    # 测试按钮信息
    assert 'A' in buttons
    assert 'B' in buttons

def test_get_action_delta(oculus_interface):
    """
    测试 get_action_delta() 方法能否正确计算增量并返回最新按钮信息
    """
    # 先调用一次，以便填充 buffer
    oculus_interface.get_action()
    # 再次调用，查看增量
    delta_action, buttons = oculus_interface.get_action_delta()
    # delta_action 应该为 shape=(2,7)
    assert delta_action.shape == (2,7)
    # 测试增量按钮
    assert 'A' in buttons
    assert 'B' in buttons