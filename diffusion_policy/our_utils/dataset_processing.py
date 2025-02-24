# data_processing.py
'''
Read before running the code:
- This file contains functions for processing the dataset collected by ourself.
- You can use ceratin functions to process your dataset if necessary.
- Please take care of the saving directory name incase of overwriting the existing data.
- You can copy the data first and then process it to avoid any data loss.

Assume you have a replay buffer saved in zarr format like this:
./Your dataset_dir
├── replay_buffer.zarr
│   ├── data
│   │   ├── action *
│   │   │   ├── 0.0
│   │   │   ├── 1.0
│   │   │   ├── ...
│   │   ├── robot_eef_pose *
│   │   │   ├── 0.0
│   │   │   ├── 1.0
│   │   │   ├── ...
│   │   ├── robot_eef_pose_vel
│   │   ├── robot_joint
│   │   ├── robot_joint_vel
│   │   ├── robot_gripper_qpos
│   │   ├── stage
│   │   └── timestamp *
│   └── meta
│       └── episode_ends
└── videos
    ├── 0
        ├── 0.mp4
        ├── 1.mp4 *
        ├── 2.mp4
        ├── 3.mp4 *
        ├── 4.mp4
    ├── 1
    ├── ...
'''

import zarr
import numpy as np
import os
import shutil
from collections import defaultdict
import click
import argparse
import cv2


# 1. Load and Save
#----------------------------------------------------------------#
def load_zarr(replay_buffer_path):
    'load all the data from the replay buffer zarr file'
    replay_buffer = zarr.open(replay_buffer_path, mode='r')
    data = {
        'actions': replay_buffer['data/action'][:],
        'robot_eef_pose': replay_buffer['data/robot_eef_pose'][:],
        'timestamps': replay_buffer['data/timestamp'][:],
        'episode_ends': replay_buffer['meta/episode_ends'][:],
        'stage': replay_buffer['data/stage'][:],
        'magnet_state': replay_buffer['data/magnet_state'][:],
        # chunks
        'actions_chunks': replay_buffer['data/action'].chunks,
        'robot_eef_pose_chunks': replay_buffer['data/robot_eef_pose'].chunks,
        'timestamps_chunks': replay_buffer['data/timestamp'].chunks,
        'episode_ends_chunks': replay_buffer['meta/episode_ends'].chunks,
        'stage_chunks': replay_buffer['data/stage'].chunks,
        'magnet_state_chunks': replay_buffer['data/magnet_state'].chunks,
    }
    return data

def load_data_with_info(base_path):
    'load all the data from the replay buffer zarr file'
    replay_buffer_path = os.path.join(base_path, 'replay_buffer.zarr')
    video_path = os.path.join(base_path, 'videos')
    replay_buffer = zarr.open(replay_buffer_path, mode='r')
    actions = replay_buffer['data/action'][:]
    robot_eef_pose = replay_buffer['data/robot_eef_pose'][:]
    timestamps = replay_buffer['data/timestamp'][:]
    episode_ends = replay_buffer['meta/episode_ends'][:]
    stage = replay_buffer['data/stage'][:]
    magnet_state = replay_buffer['data/magnet_state'][:] 
    # chunks
    actions_chunks = replay_buffer['data/action'].chunks
    robot_eef_pose_chunks = replay_buffer['data/robot_eef_pose'].chunks
    timestamps_chunks = replay_buffer['data/timestamp'].chunks
    episode_ends_chunks = replay_buffer['meta/episode_ends'].chunks
    stage_chunks = replay_buffer['data/stage'].chunks
    magnet_state_chunks = replay_buffer['data/magnet_state'].chunks
    # more info
    episode_length = episode_ends[-1]
    video_folders = sorted([int(f) for f in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, f)) and f.isdigit()])
    # get the video lengths of each mp4 files
    video_lengths = defaultdict(list)
    for folder in video_folders:
        folder_path = os.path.join(video_path, str(folder))
        video_files = os.path.join(folder_path,'1.mp4')
        cap = cv2.VideoCapture(video_files)
        if not cap.isOpened():
            print(f"Error opening video file {video_files}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:  # Check for valid fps value
            duration = frame_count / fps
            video_lengths[folder].append(duration)
        else:
            print(f"Invalid fps for video file {video_files}")
        cap.release()
    # get the average video length
    average_video_length = np.mean([np.mean(video_lengths[folder]) for folder in video_folders])
    data = {
        'actions': actions,
        'robot_eef_pose': robot_eef_pose,
        'timestamps': timestamps,
        'episode_ends': episode_ends,
        'stage': stage,
        'magnet_state': magnet_state,
        # chunks
        'actions_chunks': actions_chunks,
        'robot_eef_pose_chunks': robot_eef_pose_chunks,
        'timestamps_chunks': timestamps_chunks,
        'episode_ends_chunks': episode_ends_chunks,
        'stage_chunks': stage_chunks,
        'magnet_state_chunks': magnet_state_chunks,
        # more info
        'episode_length': episode_length,
        'video_lengths': video_lengths,
        'average_video_length': average_video_length
    }
    print('Data loaded successfully.')
    print(f"Episode length: {episode_length}")
    print(f"Average video length: {average_video_length}")
    return data

def load_zarr_based_on_index(replay_buffer_path, index):
    'index: The index of the episode to load'
    'Only load the data of the episode with the given index'
    data = load_zarr(replay_buffer_path)
    this_episode_ends = data['episode_ends'][index]
    this_episode_start = 0 if index == 0 else data['episode_ends'][index - 1] + 1
    this_episode_length = this_episode_ends - this_episode_start + 1
    this_data = {
        'actions': data['actions'][this_episode_start:this_episode_ends + 1],
        'robot_eef_pose': data['robot_eef_pose'][this_episode_start:this_episode_ends + 1],
        'timestamps': data['timestamps'][this_episode_start:this_episode_ends + 1],
        'episode_ends': data['episode_ends'][index:index + 1],
        'magnet_state': data['magnet_state'][this_episode_start:this_episode_ends + 1],
        'stage': data['stage'][this_episode_start:this_episode_ends + 1],

    }
    return this_data

def save_zarr(path, data):
    'save the data to the given replay buffer zarr file path'
    print('You are saving to the following path:', path)
    input('Press Enter to continue...')
    replay_buffer = zarr.open(path, mode='w')
    replay_buffer.create_dataset('data/action', data = data['actions'], chunks=data['actions_chunks'])
    replay_buffer.create_dataset('data/robot_eef_pose', data = data['robot_eef_pose'], chunks=data['robot_eef_pose_chunks'])
    replay_buffer.create_dataset('data/timestamp', data = data['timestamps'], chunks=data['timestamps_chunks'])
    replay_buffer.create_dataset('data/stage', data = data['stage'], chunks=data['stage_chunks'])
    replay_buffer.create_dataset('meta/episode_ends', data = data['episode_ends'], chunks=data['episode_ends_chunks'])
    replay_buffer.create_dataset('data/magnet_state', data = data['magnet_state'], chunks=data['magnet_state_chunks'])
    print('Data saved successfully.')
# 2. Add and Delete
#----------------------------------------------------------------#
def merge_datasets(data_01, data_02):
    'merge two datasets(load from load_zarr() ) into one dataset'
    # determine which is the first dataset
    if data_01['timestamps'][0] < data_02['timestamps'][0]:
        first_data = data_01
        second_data = data_02
    else:
        first_data = data_02
        second_data = data_01
    assert first_data['timestamps'][-1] < second_data['timestamps'][0]
    # merge the data
    merged_actions = np.concatenate((first_data['actions'], second_data['actions']), axis=0)
    merged_robot_eef_pose = np.concatenate((first_data['robot_eef_pose'], second_data['robot_eef_pose']), axis=0)
    merged_timestamps = np.concatenate((first_data['timestamps'], second_data['timestamps']), axis=0)
    merged_stage = np.concatenate((first_data['stage'], second_data['stage']), axis=0)
    merged_maget_state = np.concatenate((first_data['magnet_state'], second_data['magnet_state']), axis=0)
    # update episode_ends
    offset = len(first_data['actions'])
    updated_episode_ends = np.concatenate((first_data['episode_ends'], second_data['episode_ends'] + offset))
    # create the merged data
    merged_data = {
        'actions': merged_actions,
        'robot_eef_pose': merged_robot_eef_pose,
        'timestamps': merged_timestamps,
        'episode_ends': updated_episode_ends,
        'stage': merged_stage,
        'magnet_state': merged_maget_state,
        # chunks
        'actions_chunks': first_data['actions_chunks'],
        'robot_eef_pose_chunks': first_data['robot_eef_pose_chunks'],
        'timestamps_chunks': first_data['timestamps_chunks'],
        'episode_ends_chunks': first_data['episode_ends_chunks'],
        'stage_chunks': first_data['stage_chunks'],
        'magnet_state_chunks': first_data['magnet_state_chunks']
    }
    return merged_data

def delete_episode(data, index):
    'delete the episode with the given index from the dataset'
    this_episode_ends = data['episode_ends'][index]
    this_episode_start = 0 if index == 0 else data['episode_ends'][index - 1]
    print('data[actions].shape:', data['actions'].shape)
    print('len(data[episode_ends]):', len(data['episode_ends']))
    print('episode_ends:', data['episode_ends'])
    # delete the data
    data['actions'] = np.delete(data['actions'], range(this_episode_start, this_episode_ends), axis=0)
    data['robot_eef_pose'] = np.delete(data['robot_eef_pose'], range(this_episode_start, this_episode_ends), axis=0)
    data['timestamps'] = np.delete(data['timestamps'], range(this_episode_start, this_episode_ends), axis=0)
    data['stage'] = np.delete(data['stage'], range(this_episode_start, this_episode_ends), axis=0)
    data['magnet_state'] = np.delete(data['magnet_state'], range(this_episode_start, this_episode_ends), axis=0)
    
    # update episode_ends
    data['episode_ends'] = np.delete(data['episode_ends'], index)
    if index < len(data['episode_ends']):
        for i in range(index, len(data['episode_ends'])):
            data['episode_ends'][i] -= this_episode_length  
    print('data[actions].shape:', data['actions'].shape)
    print('len(data[episode_ends]):', len(data['episode_ends']))
    print('episode_ends:', data['episode_ends'])                                   
    return data

def delete_episodes(data, indices):
    'delete the episodes with the given indices from the dataset'
    for index in sorted(indices, reverse=True):
        data = delete_episode(data, index)
    return data

def add_episode(data, new_data):
    'Add a new episode to the dataset in the correct position based on timestamps'
    new_start_timestamp = new_data['timestamps'][0]
    new_end_timestamp = new_data['timestamps'][-1]
    
    # Find the correct insertion point
    insert_index = np.searchsorted(data['timestamps'], new_start_timestamp)
    
    # Split existing data arrays at the insertion point
    actions_before = data['actions'][:insert_index]
    actions_after = data['actions'][insert_index:]
    
    robot_eef_pose_before = data['robot_eef_pose'][:insert_index]
    robot_eef_pose_after = data['robot_eef_pose'][insert_index:]
    
    timestamps_before = data['timestamps'][:insert_index]
    timestamps_after = data['timestamps'][insert_index:]
    
    stage_before = data['stage'][:insert_index]
    stage_after = data['stage'][insert_index:]
    
    # Concatenate new data with the existing data
    merged_actions = np.concatenate((actions_before, new_data['actions'], actions_after), axis=0)
    merged_robot_eef_pose = np.concatenate((robot_eef_pose_before, new_data['robot_eef_pose'], robot_eef_pose_after), axis=0)
    merged_timestamps = np.concatenate((timestamps_before, new_data['timestamps'], timestamps_after), axis=0)
    merged_maget_state = np.concatenate((data['magnet_state'], new_data['magnet_state']), axis=0)
    merged_stage = np.concatenate((stage_before, new_data['stage'], stage_after), axis=0)
    
    # Update episode_ends
    new_episode_end = len(merged_actions) - 1
    offset = len(new_data['actions'])
    updated_episode_ends = []
    for end in data['episode_ends']:
        if end < insert_index:
            updated_episode_ends.append(end)
        else:
            updated_episode_ends.append(end + offset)
    updated_episode_ends.append(new_episode_end)
    
    # Create the merged data
    merged_data = {
        'actions': merged_actions,
        'robot_eef_pose': merged_robot_eef_pose,
        'timestamps': merged_timestamps,
        'episode_ends': np.array(updated_episode_ends),
        'stage': merged_stage,
        'magnet_state': merged_maget_state,
        # chunks
        'actions_chunks': data['actions_chunks'],
        'robot_eef_pose_chunks': data['robot_eef_pose_chunks'],
        'timestamps_chunks': data['timestamps_chunks'],
        'episode_ends_chunks': data['episode_ends_chunks'],
        'stage_chunks': data['stage_chunks'],
        'maget_state_chunks': data['magnet_state_chunks']
    }
    
    return merged_data
# 3. Fix certain issues
#----------------------------------------------------------------#
def fix_missing_gripper_and_6to7(data): # Temporarily used until the issue is fixed in the data collection
    'Fix the missing gripper qpos issue in the dataset'
    'The data collected by DP-3d_v1.2 should not use this function'
    # 处理 action 数据
    action = data['actions']
    robot_eef_pose = data['robot_eef_pose']
    timestamps = data['timestamps']
    episode_ends = data['episode_ends']
    robot_eef_pose_vel = data['robot_eef_pose_vel']
    stage = data['stage']
    actions_chunks = data['actions_chunks']
    robot_eef_pose_chunks = data['robot_eef_pose_chunks']
    timestamps_chunks = data['timestamps_chunks']
    episode_ends_chunks = data['episode_ends_chunks']
    robot_eef_pose_vel_chunks = data['robot_eef_pose_vel_chunks']
    stage_chunks = data['stage_chunks']

    new_actions = np.zeros((action.shape[0], 7))
    new_actions[:, :3] = action[:, :3]
    new_actions[:, 4:6] = action[:, 4:6]
    new_actions[:, 6] = action[:, 3]
    new_actions[:, 3] = 0

    # 处理 gripper 数据
    robot_gripper_qpos = np.tile(robot_eef_pose[:, 3].reshape(-1, 1), (1, 2))

    # 处理 robot_eef_pose 数据
    new_robot_eef_pose = np.zeros((robot_eef_pose.shape[0], 7))
    new_robot_eef_pose[:, :3] = robot_eef_pose[:, :3]
    new_robot_eef_pose[:, 4:6] = robot_eef_pose[:, 4:6]
    new_robot_eef_pose[:, 6] = robot_eef_pose[:, 3]

    new_data = {
        'actions': new_actions,
        'robot_eef_pose': new_robot_eef_pose,
        'timestamps': timestamps,
        'episode_ends': episode_ends,
        'robot_eef_pose_vel': robot_eef_pose_vel,
        'stage': stage,
        'robot_gripper_qpos': robot_gripper_qpos,
        # chunks
        'actions_chunks': actions_chunks,
        'robot_eef_pose_chunks': robot_eef_pose_chunks,
        'timestamps_chunks': timestamps_chunks,
        'episode_ends_chunks': episode_ends_chunks,
        'robot_eef_pose_vel_chunks': robot_eef_pose_vel_chunks,
        'stage_chunks': stage_chunks,
        'robot_gripper_qpos_chunks': robot_eef_pose_chunks
    }
    return new_data

def fix_gripper_qpos_missing(data):
    'Fix the missing gripper qpos issue in the dataset'
    robot_gripper_qpos = np.tile(data['robot_eef_pose'][:, 6].reshape(-1, 1), (1, 2))
    data['robot_gripper_qpos'] = robot_gripper_qpos
    return data

def fix_seedrobot_datashift(data):
    # 比较action和robot_eef_pose，如果对应的robot_eef_pose和action相差比较大，把robot_eef_pose对应的action
    threshold = 0.05  # 定义一个阈值，当 x 和 y 之间的差异超过这个值时进行更正
    corrected_robot_eef_pose = data['robot_eef_pose'].copy()
    index_to_video = np.zeros(len(data['actions']), dtype=int)
    start_index = 0

    for idx, end in enumerate(data['episode_ends']):
        index_to_video[start_index:end + 1] = idx
        start_index = end + 1
    # 记录被替换的次数
    replacement_count = 0
    for i in range(len(data['robot_eef_pose'])):
        # 检查 x 和 y 的差异
        if abs(data['robot_eef_pose'][i, 0] - data['actions'][i, 0]) > threshold or abs(data['robot_eef_pose'][i, 1] - data['actions'][i, 1]) > threshold:
            # 打印被替换的 robot_eef_pose 和替换后的 action 以及对应的视频文件路径
            print(f"Index {i}: robot_eef_pose {data['robot_eef_pose'][i]} replaced with action {data['actions']} in episode {index_to_video[i]}")
            # 如果差异较大，将 robot_eef_pose 替换为对应的 action
            corrected_robot_eef_pose[i, 0] = data['actions'][i, 0]
            corrected_robot_eef_pose[i, 1] = data['actions'][i, 1]
            replacement_count += 1
    # 计算被替换的比例
    total_count = len(data['robot_eef_pose'])
    replacement_percentage = (replacement_count / total_count) * 100
    print(f"Total replacements: {replacement_count} out of {total_count} ({replacement_percentage:.2f}%)")
    data['robot_eef_pose'] = corrected_robot_eef_pose
    return data

def fix_missing_additional_info(data,index):
    'Fix the missing additional info issue in the dataset'
    data['additional_info'] = np.full((data['actions'].shape[0], 1), index)
    # data['additional_info'] = np.full((data['actions'].shape[0], 1), 1)
    # data['additional_info'] = np.full((data['actions'].shape[0], 1), 2)
    return data

def fix_umi_data_missing(data):
    """
    Fix the missing additional info issue in the UMI dataset.
    
    The function assumes that the UMI dataset contains:
    - Camera observations (e.g., camera0_rgb) with specific latency and down-sample steps.
    - Robot end-effector pose, rotation, and gripper information with respective latency and down-sample steps.
    """
    # Default parameters based on the provided task info
    camera_obs_latency = 0.125
    robot_obs_latency = 0.0001
    gripper_obs_latency = 0.02
    dataset_frequency = 59.94
    obs_down_sample_steps = 3

    # Recalculate latency steps based on the given formulae
    camera_to_robot_latency_steps = int((camera_obs_latency - robot_obs_latency) * dataset_frequency)
    camera_to_gripper_latency_steps = int((camera_obs_latency - gripper_obs_latency) * dataset_frequency)
    
    # Ensure proper alignment of data (you can refine this logic based on the dataset structure)
    if 'robot_eef_pos' in data and 'robot_gripper_qpos' not in data:
        # If gripper qpos data is missing, infer it using robot eef pose (based on your task configuration)
        data['robot_gripper_qpos'] = np.tile(data['robot_eef_pose'][:, 6].reshape(-1, 1), (1, 2))

    # Fixing down-sampling and aligning with horizons
    obs_horizon = 2
    action_horizon = 16

    # Adding any other missing fields that are crucial based on the task structure
    if 'camera0_rgb' not in data:
        data['camera0_rgb'] = np.zeros((data['actions'].shape[0], 3, 224, 224))  # Assuming a default shape of [3, 224, 224]

    if 'robot0_eef_rot_axis_angle' not in data:
        data['robot0_eef_rot_axis_angle'] = np.zeros((data['actions'].shape[0], 6))  # Assuming rotation in 6D as described

    # Adjust the data based on latency steps
    data['robot0_eef_pos'] = np.roll(data['robot_eef_pose'], shift=-camera_to_robot_latency_steps, axis=0)
    data['robot0_gripper_width'] = np.roll(data['robot_gripper_qpos'], shift=-camera_to_gripper_latency_steps, axis=0)

    # Handle any other missing additional info fields specific to UMI dataset
    if 'additional_info' not in data:
        data['additional_info'] = np.zeros((data['actions'].shape[0], 1))  # Assuming a single-column additional info

    print("UMI dataset missing data fixed.")
    return data

# 4. Fix video issues
#---------------------------------------------------------------#
def video_check(base_path):
    video_folder_path = os.path.join(base_path, 'videos')
    video_folders = sorted([int(f) for f in os.listdir(video_folder_path) if os.path.isdir(os.path.join(video_folder_path, f)) and f.isdigit()])
    # check if the video file is not broken
    for folder in video_folders:
        folder_path = os.path.join(video_folder_path, str(folder))
        video_files = os.listdir(folder_path)
        if len(video_files) == 0:
            print(f"No video files found in folder {folder_path}")
            continue
        else:
            for video_file in video_files:
                video_file_path = os.path.join(folder_path, video_file)
                cap = cv2.VideoCapture(video_file_path)
                assert cap.isOpened(), f"Error opening video file {video_file_path}"
                cap.release()
    print('All video files are checked successfully.')



def rename_video_folders(base_path, max_index=179):
    'When you use delete_zarr_episode() to delete an episode, you can delete certain video folders and call this function to rename the remaining folders.'
    'The max_index is : When everything is done, the last index of the video folders, so you might need to calculate it.'
    'Also copy the videos before calling this function'
    current_folders = sorted([int(f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()])
    missing_folders = [i for i in range(max_index + 1) if i not in current_folders]
    temp_folder = os.path.join(base_path, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    for missing in reversed(missing_folders):
        for i in range(max(current_folders), missing, -1):
            old_path = os.path.join(base_path, str(i))
            new_path = os.path.join(base_path, str(i - 1))
            temp_path = os.path.join(temp_folder, str(i - 1))
            if os.path.exists(old_path):
                shutil.move(old_path, temp_path)
                print(f"Moved {old_path} to {temp_path}")
        for i in range(missing + 1, max(current_folders) + 1):
            temp_path = os.path.join(temp_folder, str(i - 1))
            new_path = os.path.join(base_path, str(i - 1))
            if os.path.exists(temp_path):
                shutil.move(temp_path, new_path)
                print(f"Moved {temp_path} to {new_path}")
        current_folders = sorted([int(f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()])
    shutil.rmtree(temp_folder)
    print('All folders renamed successfully.')


def rename_videos(dataset_path): # will be dropped in the future
    'If you happened to save the videos as 0.mp4 and 1.mp4 and want to rename them to 1.mp4 and 3.mp4, you can use this function.'
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            if '3.mp4' in files:
                print(f"Skipping folder (already named correctly): {folder_path}")
                continue
            rename_map = {}
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                if file_name == '0.mp4':
                    rename_map[file_path] = os.path.join(folder_path, '1.mp4')
                elif file_name == '1.mp4':
                    rename_map[file_path] = os.path.join(folder_path, '3.mp4')
            for old_name, new_name in rename_map.items():
                os.rename(old_name, new_name + '.tmp')
            for old_name, new_name in rename_map.items():
                os.rename(new_name + '.tmp', new_name)
            print(f"Processed folder: {folder_path}")

# 5. Operations between Datasets
#---------------------------------------------------------------#
def combine_two(base_path_1,base_path_2,save_path):
    'Combine two datasets into one dataset to train a multi-task model'
    'They are combined one dataset by one dataset, based on the time they were collected'
    zarr_file_path_1 = os.path.join(base_path_1, 'replay_buffer.zarr')
    zarr_file_path_2 = os.path.join(base_path_2, 'replay_buffer.zarr')
    video_folder_1 = os.path.join(base_path_1, 'videos')
    video_folder_2 = os.path.join(base_path_2, 'videos')
    data_1 = load_zarr(zarr_file_path_1)
    data_2 = load_zarr(zarr_file_path_2)
    # combine the data
    if data_1['timestamps'][-1] > data_2['timestamps'][-1]:
        data_tmp = data_1
        data_1 = data_2
        data_2 = data_tmp
        video_folder_tmp = video_folder_1
        video_folder_1 = video_folder_2
        video_folder_2 = video_folder_tmp
    input(f'Merging {base_path_1} and {base_path_2}, press Enter to continue...')
    merged_data = merge_datasets(data_1, data_2)
    # deal with the video files
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    merged_video_folder_path = os.path.join(save_path, 'videos')
    if not os.path.exists(merged_video_folder_path):
        os.makedirs(merged_video_folder_path)
    merged_zaar_file_path = os.path.join(save_path, 'replay_buffer.zarr')
    if not os.path.exists(merged_zaar_file_path):
        os.makedirs(merged_zaar_file_path)

    # save the merged data
    save_zarr(merged_zaar_file_path, merged_data)
    # Copy video files from video_folder_1
    for folder_name in os.listdir(video_folder_1):
        src_folder = os.path.join(video_folder_1, folder_name)
        dst_folder = os.path.join(merged_video_folder_path, folder_name)
        shutil.copytree(src_folder, dst_folder)
    
    # Copy and rename video files from video_folder_2
    offset = len(os.listdir(video_folder_1))
    for folder_name in os.listdir(video_folder_2):
        src_folder = os.path.join(video_folder_2, folder_name)
        new_folder_name = str(int(folder_name) + offset)
        dst_folder = os.path.join(merged_video_folder_path, new_folder_name)
        shutil.copytree(src_folder, dst_folder)
    print('Datasets combined successfully.')
    print(f"Saved to {save_path}")



def get_subset(base_path, start_episode, end_episode, save_path):
    zarr_file_path = os.path.join(base_path, 'replay_buffer.zarr')
    video_folder = os.path.join(base_path, 'videos')
    data = load_zarr(zarr_file_path)
    # get the subset of the data according to start_index and end_index
    start_index = data['episode_ends'][start_episode - 1] if start_episode > 0 else 0
    end_index = data['episode_ends'][end_episode]
    # subset = data[start_index:end_index + 1]
    subset = {
        'actions': data['actions'][start_index:end_index + 1],
        'robot_eef_pose': data['robot_eef_pose'][start_index:end_index + 1],
        'timestamps': data['timestamps'][start_index:end_index + 1],
        'episode_ends': data['episode_ends'][start_episode:end_episode + 1],
        'stage': data['stage'][start_index:end_index + 1],
        'maget_state': data['magnet_state'][start_index:end_index + 1],
        # chunks
        'actions_chunks': data['actions_chunks'],
        'robot_eef_pose_chunks': data['robot_eef_pose_chunks'],
        'timestamps_chunks': data['timestamps_chunks'],
        'episode_ends_chunks': data['episode_ends_chunks'],
        'stage_chunks': data['stage_chunks'],
        'maget_state_chunks': data['magnet_state_chunks']
    }
    # deal with video files
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    subset_video_folder_path = os.path.join(save_path, 'videos')
    if not os.path.exists(subset_video_folder_path):
        os.makedirs(subset_video_folder_path)
    subset_zaar_file_path = os.path.join(save_path, 'replay_buffer.zarr')
    if not os.path.exists(subset_zaar_file_path):
        os.makedirs(subset_zaar_file_path)
    # save the subset data
    # for video in video_foler: if video in the subset, copy it to the subset_video_folder_path
    for folder_name in os.listdir(video_folder):
        folder_index = int(folder_name)
        if folder_index >= start_episode and folder_index <= end_episode:
            src_folder = os.path.join(video_folder, folder_name)
            dst_folder = os.path.join(subset_video_folder_path, folder_name)
            shutil.copytree(src_folder, dst_folder)
    # save zarr
    save_zarr(subset_zaar_file_path, subset)

def length_check(base_path):
    zarr_file_path = os.path.join(base_path, 'replay_buffer.zarr')
    data = load_zarr(zarr_file_path)
    print(f"data['actions'].len = {len(data['actions'])}")
    print(f"data['robot_eef_pose'].len = {len(data['robot_eef_pose'])}")
    print(f"data['timestamps'].len = {len(data['timestamps'])}")
    print(f"data['episode_ends'].len = {len(data['episode_ends'])}")
    print(f"episode_ends = {data['episode_ends']}")
    print(f"data['robot_eef_pose_vel'].len = {len(data['robot_eef_pose_vel'])}")
    print(f"data['stage'].len = {len(data['stage'])}")

# 6. Process the dataset by runing the following codes:
#---------------------------------------------------------------#

def main():
    parser = argparse.ArgumentParser(description="dataset processing")
    parser.add_argument('--data-file', '-p' ,type=str, required=True, help="Path to the input data file (npz format).")
    parser.add_argument('--output-file', '-o' ,type=str, required=False, help="Path to the output data file (npz format).")
    parser.add_argument('--data-file-2', '-p2' ,type=str, required=False, help="Path to the input data file 2 (npz format).")
    parser.add_argument('--flag', '-f' ,type=int, required=False, help="The flag to indicate the operation you want to perform.")


    args = parser.parse_args()

    flag = args.flag if args.flag else 0

    if flag==1: # dataset collected by  dp-3d_v1.2 or dp-3d_v2.0
        data = load_zarr(args.data_file)
        data = fix_seedrobot_datashift(data)
        data = fix_gripper_qpos_missing(data)
        if args.output_file:
            save_zarr(args.output_file, data)
        else:
            # ../origin_replay_buffer.zarr->../replay_buffer.zarr
            path = args.data_file.split('/')
            path[-1] = 'replay_buffer.zarr'
            save_zarr('/'.join(path), data)
    elif flag==2: # dataset collected by dp-3d
        data = load_zarr(args.data_file)
        data = fix_missing_gripper_and_6to7(data)
        data = fix_seedrobot_datashift(data)
        if args.output_file:
            save_zarr(args.output_file, data)
        else:
            # ../origin_replay_buffer.zarr->../replay_buffer.zarr
            path = args.data_file.split('/')
            path[-1] = 'replay_buffer.zarr'
            save_zarr('/'.join(path), data)
    elif flag==3: # to handle certain issue 
        #+---------------------------------+
        #|             camera missing       |
        missing_index = 56
        max_index = 55
        base_path = args.data_file
        zarr_file = os.path.join(base_path, 'replay_buffer.zarr')
        print('zapath:', zarr_file)
        data = load_zarr(zarr_file)
        data = delete_episode(data, missing_index)
        save_zarr(zarr_file, data)
        # rename_video_folders(os.path.join(base_path, 'videos'), max_index)
        pass
    elif flag==4: # to handle certain issue
        #+---------------------------------+
        #|              data combine       |
        base_path_1 = args.data_file
        base_path_2 = args.data_file_2
        save_path = args.output_file
        combine_two(base_path_1, base_path_2, save_path)
        '''
        python dataset_processing.py -p /home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/dataset_basketball -p2 /home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/dataset_ringtoss -o /home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/dataset_basketball_ringtoss
        '''
    elif flag==5: # to handle certain issue
        #+---------------------------------+
        #|              get subset         |
        base_path = args.data_file
        start_episode = 0
        end_episode = 10
        save_path = args.output_file
        get_subset(base_path, start_episode, end_episode, save_path)
        '''
        python dataset_processing.py -p /home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/dataset_cupstack -o /home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/dataset_test
        '''
    elif flag==6: # to handle certain issue
        #+---------------------------------+
        #|              more info         |
        base_path = args.data_file
        load_data_with_info(base_path)
        '''
        python dataset_processing.py -p /home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/dataset_fold_0711
        '''
    else: # debug
        data = load_zarr(args.data_file)
        data = fix_seedrobot_datashift(data)
        data = fix_gripper_qpos_missing(data)
        data = fix_missing_additional_info(data, 3)
        save_zarr(args.output_file, data)
        pass
        '''
        python dataset_processing.py -p /home/zcai/workspaces/hzy_space/dp/dp-3d-1.2/data/blockpick_multiview
        '''
    print('Done.')
if __name__ == "__main__":
    main()

# python dataset_processing.py -p /home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/dataset_fold_0711