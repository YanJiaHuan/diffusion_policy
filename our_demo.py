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



# Launch Hyperparameters
@click.command()
@click.option('--checkpoint_path', '-m', required = True, type=str, help='Path to the model checkpoint')
@click.option('--can_interface', '-c', default='can_piper', help="CAN interface to use.")
@click.option('--output_dir', '-o', type=str, default ='./data/real_test', help='Output directory')
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
                        print('Inference latency:', time.time() - s)


                    # deal with timing
                    # the same step actions are always the target for
                    action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
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
                    print('send_actions', action)
                    # make the last dim of each action in action to be 1.0
                    env.exec_actions(
                        actions=action,
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