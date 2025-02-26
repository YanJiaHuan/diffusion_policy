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
from diffusion_policy.encapsulated_oculusReader.oculus_reader import OculusReader
from diffusion_policy.encapsulated_oculusReader.oculus_data import OculusHandler
from scipy.spatial.transform import Rotation as R

import math

@click.command()
@click.option('--output', '-o', default = 'data/our_collected_data/pickplace_test_2', help="Directory to save demonstration dataset.")
@click.option('--can_interface', '-c', default='can_piper', help="CAN interface to use.")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--reset', '-r', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, can_interface, vis_camera_idx, reset, frequency, command_latency):

    dt = 1/frequency
    ocu_hz = 60
    oculus_reader = OculusReader()
    handler = OculusHandler(oculus_reader, right_controller=True, hz=ocu_hz, use_filter=False, max_threshold=0.6/ocu_hz)
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
                shm_manager=shm_manager,
                reset = reset
            ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)
            time.sleep(1.0)


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
                increment = handler.get_increment()
                buttons = handler.get_buttons()
                original_orientation = handler.get_original_orientation()
                #-----------------旋转矩阵转欧拉角(now 旋转向量)---------------------
                r = R.from_matrix(original_orientation)
                rotation_vector = r.as_rotvec()
                # euler_angles = r.as_euler('xyz', degrees=True)
                # rpy_vr = np.array(np.mod(euler_angles + 180, 360) - 180)
                #-----------------获取VR手柄按键状态-------------------
                A_button = buttons.get("A", [0])
                B_button = buttons.get("B", [0])
                right_trigger = buttons.get("rightTrig", [0])[0]
                delta_pos = increment['position']
                #---------------------------------------------------
                # 判断是否按下A键，按下A，机械臂允许移动
                #---------------------------------------------------
                if A_button:
                    freeze = False
                else:
                    freeze = True
                #----------------------------------------------------
                #------------------更新endpose状态---------------------
                #获得当前机械臂状态
                target_pose = env.get_robot_state()['ActualTCPPose']
                # xyz in meters, roll pitch yaw in degrees
                if freeze:
                    # convert the euler into rotation vector back to robot controller
                    target_pose[0] = target_pose[0]
                    target_pose[1] = target_pose[1]
                    target_pose[2] = target_pose[2]
                    rotation_vector = R.from_euler('xyz', target_pose[3:6], degrees=True).as_rotvec()
                    target_pose[3:6] = rotation_vector
                else:
                    target_pose[0] = target_pose[0] + delta_pos[0]*0.1
                    target_pose[1] = target_pose[1] + delta_pos[1]*0.1
                    target_pose[2] = target_pose[2] + delta_pos[2]*0.1
                    # TODO: 目前VR的旋转是用绝对值，xyz是增量，其原因是VR的旋转增量没有调通
                    # target_pose[3] = np.deg2rad(rpy_vr[0])
                    # target_pose[4] = np.deg2rad(rpy_vr[1])
                    # target_pose[5] = np.deg2rad(rpy_vr[2])
                    target_pose[3:6] = rotation_vector  # 使用旋转向量更新旋转部分
                #---------------------------------------------------
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

# %%
if __name__ == '__main__':
    main()