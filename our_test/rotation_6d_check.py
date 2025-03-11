import zarr
import numpy as np
import random
import matplotlib.pyplot as plt
import os

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
    return data

def plot_rotation_6d(base_path):
    'Randomly select 3 episodes and plot action and robot_eef_pose 6D rotations'
    zarr_file_path = os.path.join(base_path, 'replay_buffer.zarr')
    data = load_zarr(zarr_file_path)

    selected_episodes = random.sample(range(len(data['episode_ends']) - 1), 3)

    for episode_idx in selected_episodes:
        start_idx = data['episode_ends'][episode_idx]
        end_idx = data['episode_ends'][episode_idx + 1]

        actions = data['actions'][start_idx:end_idx]
        robot_eef_pose = data['robot_eef_pose'][start_idx:end_idx]

        # Prepare the figure for the subplots (each plot has 6 subplots for the 6D rotation components)
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        for i, ax in enumerate(axes):
            ax.set_title(f'Episode {episode_idx} - Action and Robot Pose Rotation 6D', fontsize=14)
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Rotation Value', fontsize=12)
            ax.grid(True)

        # Plot the 6D rotation for actions and robot poses for each dimension
        for dim in range(6):
            # Action rotation 6D is in the first 6 elements (after transformation)
            ax = axes[dim % 3]  # We have 3 subplots (for 3 episodes)
            ax.plot(range(start_idx, end_idx), actions[:, dim], label=f'Action Rotation Dim {dim+1}', linestyle='-', color='blue')
            ax.plot(range(start_idx, end_idx), robot_eef_pose[:, dim], label=f'Robot Pose Rotation Dim {dim+1}', linestyle='--', color='orange')

            ax.legend(fontsize=10)

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

def main(base_path):
    print(f"Processing dataset in {base_path}")
    plot_rotation_6d(base_path)

if __name__ == "__main__":
    base_path = '/home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/test2'  # Update this to the actual base path
    main(base_path)
