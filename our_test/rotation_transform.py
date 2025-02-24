import zarr
import numpy as np
from scipy.spatial.transform import Rotation as R
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
        'chunk_sizes': {
            'action': replay_buffer['data/action'].chunks,
            'robot_eef_pose': replay_buffer['data/robot_eef_pose'].chunks,
            'timestamp': replay_buffer['data/timestamp'].chunks,
            'episode_ends': replay_buffer['meta/episode_ends'].chunks,
            'magnet_state': replay_buffer['data/magnet_state'].chunks if 'data/magnet_state' in replay_buffer else None,
        }
    }
    return data

def transform_to_rotation_6d(rotation_vector_or_euler, is_rotation_vector=True):
    'Transform rotation vector or Euler angles to 6D rotation representation'
    if is_rotation_vector:
        # Convert rotation vector to rotation matrix
        rotation_matrix = R.from_rotvec(rotation_vector_or_euler).as_matrix()
    else:
        # Convert Euler angles to rotation matrix
        rotation_matrix = R.from_euler('xyz', rotation_vector_or_euler, degrees=True).as_matrix()
    
    # Extract the first two columns of the rotation matrix for the 6D representation
    rotation_6d = rotation_matrix[:, :2].flatten()  # 6D representation is two columns of the rotation matrix
    return rotation_6d

def update_actions_and_poses(data):
    'Update actions and robot poses with 6D rotations'
    actions = data['actions']
    robot_eef_pose = data['robot_eef_pose']
    
    # Update actions (7D -> 10D)
    updated_actions = []
    for action in actions:
        position = action[:3]
        rotation_vector = action[3:6]
        magnet_command = action[6]
        
        # Transform rotation vector to 6D
        rotation_6d = transform_to_rotation_6d(rotation_vector, is_rotation_vector=True)
        
        # Update to new 10D action: position + 6D rotation + magnet command
        updated_actions.append(np.concatenate([position, rotation_6d, [magnet_command]]))
    
    updated_actions = np.array(updated_actions)

    # Update robot eef pose (6D -> 9D)
    updated_robot_eef_pose = []
    for pose in robot_eef_pose:
        position = pose[:3]
        euler_angles = pose[3:]
        
        # Transform Euler angles to 6D
        rotation_6d = transform_to_rotation_6d(euler_angles, is_rotation_vector=False)
        
        # Update to new 9D pose: position + 6D rotation
        updated_robot_eef_pose.append(np.concatenate([position, rotation_6d]))
    
    updated_robot_eef_pose = np.array(updated_robot_eef_pose)

    data['actions'] = updated_actions
    data['robot_eef_pose'] = updated_robot_eef_pose
    
    return data

def save_to_new_zarr(data, output_path, chunk_sizes):
    'Save the transformed data to a new Zarr file, using the original chunk sizes'
    root = zarr.open(output_path, mode='w')
    
    # Use the original chunk sizes
    root.create_dataset('data/action', data=data['actions'], chunks=chunk_sizes['action'])
    root.create_dataset('data/robot_eef_pose', data=data['robot_eef_pose'], chunks=chunk_sizes['robot_eef_pose'])
    root.create_dataset('data/timestamp', data=data['timestamps'], chunks=chunk_sizes['timestamp'])
    root.create_dataset('meta/episode_ends', data=data['episode_ends'], chunks=chunk_sizes['episode_ends'])
    if data['magnet_state'] is not None:
        root.create_dataset('data/magnet_state', data=data['magnet_state'], chunks=chunk_sizes['magnet_state'])
    
    print(f"New Zarr file saved at {output_path}")

def main(base_path):
    input_path = os.path.join(base_path, 'origin_replay_buffer.zarr')
    output_path = os.path.join(base_path, 'replay_buffer.zarr')
    
    print(f"Loading data from {input_path}")
    data = load_zarr(input_path)
    
    print("Transforming actions and robot poses...")
    updated_data = update_actions_and_poses(data)
    
    print("Saving the transformed data...")
    save_to_new_zarr(updated_data, output_path, data['chunk_sizes'])
    
    print("Transformation and saving completed.")

if __name__ == "__main__":
    base_path = '/home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/pickplace_v2'  # Update this to the actual base path
    main(base_path)
