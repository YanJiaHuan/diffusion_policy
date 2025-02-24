import zarr
import numpy as np
import os
import shutil
import cv2
import argparse

# Function to load data from the zarr file
def load_zarr(replay_buffer_path):
    'Load all the data from the replay buffer zarr file'
    replay_buffer = zarr.open(replay_buffer_path, mode='r')
    data = {
        'actions': replay_buffer['data/action'][:],
        'robot_eef_pose': replay_buffer['data/robot_eef_pose'][:],
        'timestamps': replay_buffer['data/timestamp'][:],
        'episode_ends': replay_buffer['meta/episode_ends'][:],
        'magnet_state': replay_buffer['data/magnet_state'][:] if 'data/magnet_state' in replay_buffer else None,
        'stage': replay_buffer['data/stage'][:],
    }
    return data

# Function to delete an episode from the dataset
def delete_episode(data, index):
    'delete the episode with the given index from the dataset'
    this_episode_ends = data['episode_ends'][index]
    this_episode_start = 0 if index == 0 else data['episode_ends'][index - 1] + 1
    this_episode_length = this_episode_ends - this_episode_start + 1
    
    # delete the data
    data['actions'] = np.delete(data['actions'], range(this_episode_start, this_episode_ends + 1), axis=0)
    data['robot_eef_pose'] = np.delete(data['robot_eef_pose'], range(this_episode_start, this_episode_ends + 1), axis=0)
    data['timestamps'] = np.delete(data['timestamps'], range(this_episode_start, this_episode_ends + 1), axis=0)
    data['robot_eef_pose_vel'] = np.delete(data['robot_eef_pose_vel'], range(this_episode_start, this_episode_ends + 1), axis=0)
    data['stage'] = np.delete(data['stage'], range(this_episode_start, this_episode_ends + 1), axis=0)
    data['robot_gripper_qpos'] = np.delete(data['robot_gripper_qpos'], range(this_episode_start, this_episode_ends + 1), axis=0)
    
    # update episode_ends
    data['episode_ends'] = np.delete(data['episode_ends'], index)
    if index < len(data['episode_ends']):
        for i in range(index, len(data['episode_ends'])):
            data['episode_ends'][i] -= this_episode_length                                     
    return data

# Function to delete multiple episodes based on the input indices
def delete_episodes(data, indices):
    'Delete the episodes with the given indices from the dataset'
    for index in sorted(indices, reverse=True):
        data = delete_episode(data, index)
    return data

# Function to rename the video folders
def rename_video_folders(base_path, deleted_folders):
    'Renames the video folders to ensure continuous numbering after deletion'
    current_folders = sorted([int(f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()])
    
    # Remove deleted folders from the current folder list
    current_folders = [folder for folder in current_folders if folder not in deleted_folders]
    
    # Renumber the remaining video folders
    for old_idx, new_idx in zip(current_folders, range(len(current_folders))):
        old_folder_path = os.path.join(base_path, str(old_idx))
        new_folder_path = os.path.join(base_path, str(new_idx))
        if old_folder_path != new_folder_path:
            shutil.move(old_folder_path, new_folder_path)
            print(f"Renamed folder {old_idx} to {new_idx}")

# Function to save the updated data back to the zarr file
def save_zarr(path, data):
    'Save the data to the given replay buffer zarr file path'
    print('You are saving to the following path:', path)
    input('Press Enter to continue...')
    replay_buffer = zarr.open(path, mode='w')
    replay_buffer.create_dataset('data/action', data=data['actions'])
    replay_buffer.create_dataset('data/robot_eef_pose', data=data['robot_eef_pose'])
    replay_buffer.create_dataset('data/timestamp', data=data['timestamps'])
    replay_buffer.create_dataset('data/stage', data=data['stage'])
    if data['magnet_state'] is not None:
        replay_buffer.create_dataset('data/magnet_state', data=data['magnet_state'])
    replay_buffer.create_dataset('meta/episode_ends', data=data['episode_ends'])
    print('Data saved successfully.')

def main(base_path, episodes_to_delete):
    # Load the zarr data
    zarr_file_path = os.path.join(base_path, 'origin_replay_buffer.zarr')
    data = load_zarr(zarr_file_path)

    # Delete the specified episodes
    data = delete_episodes(data, episodes_to_delete)

    # Save the updated data back to zarr
    save_zarr(zarr_file_path, data)

    # Rename the video folders to maintain continuity
    video_folder_path = os.path.join(base_path, 'videos')
    rename_video_folders(video_folder_path, episodes_to_delete)

    print(f"Deleted episodes {episodes_to_delete} and renamed video folders successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset processing for missing video folders and episodes.")
    parser.add_argument('--base-path', '-p', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--episodes', '-e', type=int, nargs='+', required=True, help="List of episodes to delete (e.g., 2 5).")
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.base_path, args.episodes)
