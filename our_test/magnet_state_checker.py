import zarr
import matplotlib.pyplot as plt
import numpy as np

# Path to your Zarr file
zarr_file_path = '/home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/clean_mark/replay_buffer.zarr'

# Open the Zarr file
zarr_store = zarr.open(zarr_file_path, mode='r')

# Access the 'episode_ends' and 'magnet_state' groups
episode_ends = zarr_store['meta/episode_ends'][:]
magnet_state_data = zarr_store['data/magnet_state'][:]

# Maximum number of subplots per figure
max_subplots_per_figure = 10

# Calculate the total number of episodes
num_episodes = len(episode_ends) - 1

# Calculate how many figures are needed
num_figures = (num_episodes // max_subplots_per_figure) + (1 if num_episodes % max_subplots_per_figure != 0 else 0)

# List to keep track of episodes where magnet state is always 0.0
non_working_episodes = []

# Iterate through each figure and create subplots for the episodes
for fig_idx in range(num_figures):
    # Create a new figure with up to 10 subplots
    fig, axes = plt.subplots(min(max_subplots_per_figure, num_episodes - fig_idx * max_subplots_per_figure), 1, figsize=(10, 6 * max_subplots_per_figure))
    
    # Ensure axes is iterable (in case there's only one episode in the figure)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    
    # Iterate through the episodes in the current figure
    for i in range(fig_idx * max_subplots_per_figure + 1, min((fig_idx + 1) * max_subplots_per_figure, num_episodes)):
        # Calculate the episode length
        start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        
        # Slice the magnet_state data to get the relevant data for the current episode
        episode_magnet_state = magnet_state_data[start_idx:end_idx]
        
        # If the magnet_state has multiple components, take the first component to plot
        # Here magnet_state is assumed to be binary (0.0 or 1.0)
        magnet_state_values = episode_magnet_state[:, 0]  # Assuming magnet_state is 2D
        
        # Check if the magnet state is always 0.0 (sensor not working)
        if np.all(magnet_state_values == 0.0):
            non_working_episodes.append(i)
            # Visualize the episode where magnet state is always 0.0
            axes[i - fig_idx * max_subplots_per_figure].step(range(start_idx, end_idx), magnet_state_values, where='post', linestyle='-', marker='o', color='red')
        else:
            # Plot the magnet state change for the current episode as discrete steps
            axes[i - fig_idx * max_subplots_per_figure].step(range(start_idx, end_idx), magnet_state_values, where='post', linestyle='-', marker='o')
        
        # Set titles and labels for each subplot
        axes[i - fig_idx * max_subplots_per_figure].set_xlabel('Timesteps')
        axes[i - fig_idx * max_subplots_per_figure].set_ylabel('Magnet State')
        axes[i - fig_idx * max_subplots_per_figure].set_title(f'Magnet State Change - Episode {i}')
        axes[i - fig_idx * max_subplots_per_figure].grid(True)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the figure
    plt.show()

# Print out the list of non-working episodes
print("Non-working episodes where the magnet state is always 0.0:")
print(non_working_episodes)
