"""
使用方法请看README.md
这是基于diffusion_policy的数据采集程序，迁移到piper机械臂，以及特定的设计需求(电磁铁控制替代夹爪控制)

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import PiperRealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.encapsulated_oculusReader.oculus_data_jh import OculusInterface
from diffusion_policy.encapsulated_oculusReader.oculus_reader import OculusReader
from scipy.spatial.transform import Rotation as R

import math

@click.command()
@click.option('--output', '-o', default = 'data/our_collected_data/pickplace_v4', help="Directory to save demonstration dataset.")
@click.option('--can_interface', '-c', default='can_piper', help="CAN interface to use.")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, can_interface, vis_camera_idx, frequency, command_latency):
    dt = 1/frequency
    oculus_interface = OculusInterface(oculus_reader=OculusReader(),degree=True,filter=True)
    magnet_state = False
    last_magnet_state = False
    bt_port='/dev/rfcomm0'
    baud_rate=115200
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            PiperRealEnv(
                output_dir=output, 
                can_interface = can_interface,
                bt_port=bt_port,
                baud_rate=baud_rate,
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                n_obs_steps=2,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)
            time.sleep(1.0)
            print('ready')

            state = env.get_robot_state()
            target_pose = state['ActualTCPPose']
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()
 
                precise_wait(t_sample)
                # get teleop command
                actions, buttons = oculus_interface.get_action_delta()
                A_button = buttons.get("A", [0])
                right_trigger = buttons.get("rightTrig", [0])[0]
                # 判断是否按下A键，按下A，机械臂允许移动
                #---------------------------------------------------
                if A_button:
                    freeze = False
                else:
                    freeze = True
                #----------------------------------------------------
                if freeze:
                    target_pose = env.get_robot_state()['ActualTCPPose']
                else:
                    actions, buttons = oculus_interface.get_action_delta()
                    action = actions[0]
                    delta_pos = action[:3]
                    delta_quat = action[3:]
                    scale = 10
                    delta_pos = delta_pos * scale
                    curr_pose = env.get_robot_state()['ActualTCPPose']
                    current_quat = R.from_rotvec(curr_pose[3:6]).as_quat()
                    new_quat = quat_multiply(current_quat, oculus_interface.quat_inverse(delta_quat))
                    # new_quat = quat_multiply(current_quat, delta_quat)
                    for i in range(scale):
                        new_quat = quat_multiply(current_quat, oculus_interface.quat_inverse(delta_quat))
                        # new_quat = quat_multiply(current_quat, delta_quat)
                        current_quat = new_quat
                    new_rot_vec = R.from_quat(new_quat).as_rotvec()
                    target_pose[0] = curr_pose[0] + delta_pos[0]
                    target_pose[1] = curr_pose[1] + delta_pos[1]
                    target_pose[2] = curr_pose[2] + delta_pos[2]
                    target_pose[3:6] = new_rot_vec
                #---------------电磁铁状态转换控制---------------------
                if right_trigger >= 0.8 and not last_magnet_state:
                    magnet_state = not magnet_state
                    last_magnet_state = True
                elif right_trigger < 0.8:
                    last_magnet_state = False
                else:
                    pass
                #---------------合并电磁铁状态和末端位姿----------------
                action_7d = np.zeros(7, dtype=np.float64)
                action_7d[:6] = target_pose[:6]
                if magnet_state:
                    magnet_on = 1.0
                else:
                    magnet_on = 0.0
                action_7d[6] = magnet_on 

                #execute teleop command
                env.exec_actions(
                    actions=[action_7d], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage]
                    )
                precise_wait(t_cycle_end)
                iter_idx += 1

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




# %%
if __name__ == '__main__':
    main()