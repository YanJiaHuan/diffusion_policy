import zarr
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_action(action_data):
    plt.figure(figsize=(15, 8))
    for i in range(action_data.shape[1]):
        plt.plot(action_data[:, i], label=f'Action Dim {i+1}')
    plt.title('Action Data Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_magnet_state(magnet_state_data):
    plt.figure(figsize=(15, 4))
    plt.plot(magnet_state_data, label='Magnet State', color='magenta')
    plt.title('Magnet State Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Magnet State')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_robot_eef_pose(robot_eef_pose_data):
    plt.figure(figsize=(15, 8))
    for i in range(robot_eef_pose_data.shape[1]):
        plt.plot(robot_eef_pose_data[:, i], label=f'EEF Pose Dim {i+1}')
    plt.title('Robot End-Effector Pose Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Pose Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_stage(stage_data):
    plt.figure(figsize=(15, 4))
    plt.step(range(len(stage_data)), stage_data, where='mid', label='Stage', color='green')
    plt.title('Stage Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Stage')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_timestamp(timestamp_data):
    plt.figure(figsize=(15, 4))
    plt.plot(timestamp_data, label='Timestamp', color='orange')
    plt.title('Timestamp Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Timestamp (s)')
    plt.legend()
    plt.grid(True)
    plt.show()

def explore_and_plot_zarr(zarr_path):
    if not os.path.exists(zarr_path):
        print(f"Zarr path does not exist: {zarr_path}")
        return
    
    zarr_store = zarr.open(zarr_path, mode='r')
    
    for key in zarr_store:
        group = zarr_store[key]
        print(f"Processing group: {key}/")
        
        for dataset_key in group:
            dataset = group[dataset_key]
            data = dataset[:]
            print(f"  Dataset: {dataset_key} | Shape: {data.shape} | Dtype: {data.dtype}")
            
            if key == 'data':
                if dataset_key == 'action':
                    plot_action(data)
                elif dataset_key == 'magnet_state':
                    plot_magnet_state(data)
                elif dataset_key == 'robot_eef_pose':
                    plot_robot_eef_pose(data)
                elif dataset_key == 'stage':
                    plot_stage(data)
                elif dataset_key == 'timestamp':
                    plot_timestamp(data)
                else:
                    print(f"    No plotting function defined for dataset: {dataset_key}")
            elif key == 'meta':
                if dataset_key == 'episode_ends':
                    print("  Meta Dataset 'episode_ends' contains:", data)
                else:
                    print(f"    No plotting function defined for meta dataset: {dataset_key}")
            else:
                print(f"    No plotting function defined for group: {key}")

# 示例用法
zarr_file_path = "/home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/clean_mark/replay_buffer.zarr"
explore_and_plot_zarr(zarr_file_path)
