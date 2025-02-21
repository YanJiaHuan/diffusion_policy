"""
使用方法请看README.md

基于diffusion_policy 的eval_real_robot.py修改，原本作者在eval的时候嵌入了人工控制机械臂的代码，
现仅保留模型控制的逻辑

"""

import time
import click
import cv2
import numpy as np
import torch
import dill
import hydra

from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.real_env import PiperRealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import get_real_obs_resolution, get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from scipy.spatial.transform import Rotation as R

# Launch Hyperparameters
@click.command()
@click.option('--checkpoint_path', '-m', required = True, type=str, help='Path to the model checkpoint')
@click.option('--can_interface', '-c', default='can_piper', help="CAN interface to use.")
@click.option('--output_dir', '-o', type=str, default ='./data/our_test/clean_mark_v2', help='Output directory')
@click.option('--frequency', '-f', type=int, default=10, help='Control frequency')
@click.option('--steps_per_inference', '-s', type=int, default=6, help='Number of steps per inference')
@click.option('--max_duration', '-d', type=int, default=1800, help='Max duration of the experiment')

# Main function
def main(checkpoint_path, output_dir, frequency, steps_per_inference, can_interface, max_duration):
    #----------------- Load Model -------------------------------#
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    #----------------- method-specific setup --------------------#
    action_offset = 0
    delta_action = False
    #----------------- Diffusion Policy -------------------------#
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device('cuda')
    policy.eval().to(device)
    #------------------ Policy Hyperparameters -------------------#
    policy.num_inference_steps = 16 
    policy.n_obs_steps = 2
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    #------------------- Setup Experiment ------------------------#
    dt = 1/frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps

    # Initialize magnet state variables
    magnet_state = False
    last_magnet_state = False


    with SharedMemoryManager() as shm_manager:
        with PiperRealEnv(
            output_dir=output_dir, 
            can_interface = can_interface,
            # recording resolution
            obs_image_resolution=obs_res,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager,
            reset = True
        ) as env:
            cv2.setNumThreads(1)
            #------------------- Setup RealSense ------------------------#
            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)  
            # env.realsense.set_contrast(contrast=50) #default 50 
            print("Waiting for realsense")
            time.sleep(1.0)
            
            #----------------- policy inference -------------------#
            try:                
                policy.reset()
                start_delay = 1.0
                eval_t_start = time.time() + start_delay
                t_start = time.monotonic() + start_delay
                env.start_episode(eval_t_start)
                # wait for 1/30 sec to get the closest frame actually
                # reduces overall latency
                frame_latency = 1/30
                precise_wait(eval_t_start - frame_latency, time_func=time.time)
                print("Started!")
                iter_idx = 0
                term_area_start_timestamp = float('inf')
                perv_target_pose = None
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                    # get obs
                    print('get_obs')
                    obs = env.get_obs()
                    #----------------------------------------------------#
                    # Change the robot_eef_pose's rotation representation from euler to rotation_6d
                    robot_eef_pose = obs['robot_eef_pose']
                    # Process each timestep separately
                    rotation_6d_list = []
                    for i in range(robot_eef_pose.shape[0]):  # Iterating over both timesteps
                        euler_angles_deg = robot_eef_pose[i, 3:]  # Get Euler angles for timestep i
                        rotation_matrix = R.from_euler('xyz', euler_angles_deg, degrees=True).as_matrix()
                        rotation_6d = rotation_matrix[:, :2].flatten()  # Extract 6D rotation (flattened)
                        rotation_6d_list.append(rotation_6d)

                    # Stack the 6D rotations for both timesteps
                    rotation_6d_array = np.stack(rotation_6d_list)

                    # Update the robot pose with the new 6D rotation representation
                    robot_eef_pose = np.hstack((robot_eef_pose[:, :3], rotation_6d_array))  # Concatenate position and rotation_6d

                    obs['robot_eef_pose'] = robot_eef_pose
                    #----------------------------------------------------#
                    obs_timestamps = obs['timestamp']
                    # run inference
                    with torch.no_grad():
                        s = time.time()
                        obs_dict_np = get_real_obs_dict(
                            env_obs=obs, shape_meta=cfg.task.shape_meta)
                        obs_dict = dict_apply(obs_dict_np, 
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                        result = policy.predict_action(obs_dict)
                        # this action starts from the first obs step
                        action = result['action'][0].detach().to('cpu').numpy()
                        inference_latency = time.time() - s
                        print('Inference latency:', inference_latency)


                    # deal with timing
                    # the same step actions are always the target for
                    action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1] + inference_latency
                    action_exec_latency = 0.01
                    curr_time = time.time()
                    is_new = action_timestamps > (curr_time + action_exec_latency)
                    if np.sum(is_new) == 0:
                        # exceeded time budget, still do something
                        action = action[[-1]]
                        # schedule on next available step
                        next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                        action_timestamp = eval_t_start + (next_step_idx) * dt
                        print('Over budget', action_timestamp - curr_time)
                        action_timestamps = np.array([action_timestamp])
                    else:
                        action = action[is_new]
                        action_timestamps = action_timestamps[is_new]

                    
                    # execute actions
                    # need to convert each action's rotation representation from rotation_6d to roation_vector(in radians)
                    new_actions = []
                    for action_idx in range(len(action)):
                        action_7d = action[action_idx]
                        
                        # Extract position, rotation_6d, and magnet_command
                        position = action_7d[:3]  # First 3 values are position
                        rotation_6d = action_7d[3:9]  # Next 6 values are rotation_6d
                        magnet_command = action_7d[9]  # Last value is the magnet command

                        # Transform 6D rotation to rotation matrix
                        rotation_matrix = np.zeros((3, 3))
                        rotation_matrix[:, :2] = rotation_6d.reshape(3, 2)
                        
                        # The third column is computed such that the matrix remains orthogonal
                        rotation_matrix[:, 2] = np.cross(rotation_matrix[:, 0], rotation_matrix[:, 1])

                        # Convert rotation matrix to rotation vector (axis-angle representation)
                        rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()

                        # Update to new 7D action: position + rotation_vector + magnet command
                        action_7d = np.concatenate([position, rotation_vector, [magnet_command]])

                        # Append the updated action to the new_actions list
                        new_actions.append(action_7d)
                        
                    # Apply magnet state filtering logic
                    for action_7d in new_actions:
                        if action_7d[6] >= 0.8 and not last_magnet_state:
                            magnet_state = not magnet_state
                            last_magnet_state = True
                        elif action_7d[6] < 0.8:
                            last_magnet_state = False
                        action_7d[6] = 1.0 if magnet_state else 0.0


                    env.exec_actions(
                        actions=new_actions,
                        timestamps=action_timestamps
                    )
                    print(f"Submitted {len(action)} steps of actions.")

                    # auto termination
                    terminate = False
                    if time.monotonic() - t_start > max_duration:
                        terminate = True
                        print("Terminated due to timeout.")

                    if terminate:
                        env.end_episode()


                    # wait for execution
                    precise_wait(t_cycle_end - frame_latency)
                    iter_idx += steps_per_inference


            except KeyboardInterrupt:
                print("stopped!")
                env.end_episode()

# %%
if __name__ == '__main__':

    main()