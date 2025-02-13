import zarr
import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R

def load_zarr(replay_buffer_path):
    'Load all the data from the replay buffer zarr file'
    replay_buffer = zarr.open(replay_buffer_path, mode='r')
    data = {
        'actions': replay_buffer['data/action'][:],
        'robot_eef_pose': replay_buffer['data/robot_eef_pose'][:],
        'timestamps': replay_buffer['data/timestamp'][:],
        'episode_ends': replay_buffer['meta/episode_ends'][:],
        'magnet_state': replay_buffer['data/magnet_state'][:] if 'data/magnet_state' in replay_buffer else None,
    }
    
    # Convert rotation vectors in radians to Euler angles in degrees
    actions = data['actions']
    rotation_vector = actions[:, 3:6]  # assuming rotation vectors are in columns 3, 4, 5
    
    # Convert to Euler angles (in degrees)
    euler_angles = R.from_rotvec(rotation_vector).as_euler('xyz', degrees=True)
    actions[:, 3:6] = euler_angles  # replace rotation vectors with Euler angles
    
    data['actions'] = actions  # update actions with the new Euler angles

    return data

def check_corrupted_videos(base_path):
    'Check if the video files are corrupted by attempting to open each video.'
    video_folder_path = os.path.join(base_path, 'videos')
    video_folders = sorted([int(f) for f in os.listdir(video_folder_path) if os.path.isdir(os.path.join(video_folder_path, f)) and f.isdigit()])
    for folder in video_folders:
        folder_path = os.path.join(video_folder_path, str(folder))
        video_files = os.listdir(folder_path)
        for video_file in video_files:
            video_file_path = os.path.join(folder_path, video_file)
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                print(f"Error opening video file {video_file_path}")
            cap.release()
    print("Video check completed.")

def calculate_average_video_length(base_path):
    'Calculate the average video length'
    video_folder_path = os.path.join(base_path, 'videos')
    video_folders = sorted([int(f) for f in os.listdir(video_folder_path) if os.path.isdir(os.path.join(video_folder_path, f)) and f.isdigit()])
    video_lengths = []
    for folder in video_folders:
        folder_path = os.path.join(video_folder_path, str(folder))
        video_files = os.path.join(folder_path, '0.mp4')
        cap = cv2.VideoCapture(video_files)
        if not cap.isOpened():
            print(f"Error opening video file {video_files}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            duration = frame_count / fps
            video_lengths.append(duration)
        cap.release()
    average_video_length = np.mean(video_lengths)
    print(f"Average video length: {average_video_length:.2f} seconds.")
    return average_video_length

def check_magnet_state(base_path):
    'Check if the magnet state is always 0.0 for any episode'
    zarr_file_path = os.path.join(base_path, 'replay_buffer.zarr')
    data = load_zarr(zarr_file_path)
    magnet_state = data['magnet_state']
    episode_ends = data['episode_ends']
    non_working_episodes = []
    for i in range(len(episode_ends) - 1):
        start_idx = episode_ends[i]
        end_idx = episode_ends[i + 1]
        episode_magnet_state = magnet_state[start_idx:end_idx]
        if np.all(episode_magnet_state == 0.0):
            non_working_episodes.append(i)
    print("Episodes with magnet state always 0.0:", non_working_episodes)
    return non_working_episodes

def plot_data_for_random_episodes(base_path):
    'Randomly select 3 episodes and plot actions and robot states'
    zarr_file_path = os.path.join(base_path, 'replay_buffer.zarr')
    data = load_zarr(zarr_file_path)
    
    selected_episodes = random.sample(range(len(data['episode_ends']) - 1), 3)
    for episode_idx in selected_episodes:
        start_idx = data['episode_ends'][episode_idx]
        end_idx = data['episode_ends'][episode_idx + 1]
        
        actions = data['actions'][start_idx:end_idx]
        robot_eef_pose = data['robot_eef_pose'][start_idx:end_idx]
        magnet_state = data['magnet_state'][start_idx:end_idx] if data['magnet_state'] is not None else np.zeros((end_idx - start_idx, 1))
        
        fig, axes = plt.subplots(7, 1, figsize=(10, 20))
        for dim in range(6):  # Only plot the first 6 dimensions for the robot state
            axes[dim].plot(range(start_idx, end_idx), actions[:, dim], label=f'Action dim {dim}', linestyle='-', color='blue')
            axes[dim].plot(range(start_idx, end_idx), robot_eef_pose[:, dim], label=f'Robot State dim {dim}', linestyle='--', color='orange')
            axes[dim].set_title(f'Dim {dim} for Episode {episode_idx}', fontsize=10)
            axes[dim].legend(fontsize=8)
            axes[dim].grid(True)
        
        # Plot magnet state as the 7th subplot
        axes[6].plot(range(start_idx, end_idx), actions[:, 6], label='Magnet Command', color='blue', linestyle='-')
        axes[6].plot(range(start_idx, end_idx), magnet_state, label='Magnet State', color='orange', linestyle='--')
        axes[6].set_title(f'Magnet State for Episode {episode_idx}', fontsize=10)
        axes[6].legend(fontsize=8)
        axes[6].grid(True)
        
        plt.tight_layout()
        plt.show()

def main(base_path):
    print(f"Processing dataset in {base_path}")
    
    # 1. Check for corrupted videos
    check_corrupted_videos(base_path)
    
    # 2. Calculate average video length
    calculate_average_video_length(base_path)
    
    # 3. Check magnet state for episodes
    check_magnet_state(base_path)
    
    # 4. Plot actions and robot states for 3 random episodes
    plot_data_for_random_episodes(base_path)
    
    print("All tasks completed.")

if __name__ == "__main__":
    base_path = '/home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/clean_mark'  # Update this to the actual base path
    main(base_path)
