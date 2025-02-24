import math
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


EPSILON = 1e-5


def plot_trajectories(ground_truth: torch.Tensor,
                      predicted: torch.Tensor,
                      epoch: int,
                      output_dir: str):
    batch_size = ground_truth.shape[0]
    sample_numbers = random.sample(range(batch_size), 3)

    # Ensure the 'plots' directory exists
    destination_dir = os.path.join(output_dir, 'plots')
    os.makedirs(destination_dir, exist_ok=True)

    for i, sample_num in enumerate(sample_numbers):
        gt_sample = ground_truth[sample_num]
        pred_sample = predicted[sample_num]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Ground Truth Trajectory
        ax.plot(gt_sample[:, 0].cpu().numpy(),
                gt_sample[:, 1].cpu().numpy(),
                gt_sample[:, 2].cpu().numpy(),
                label='Ground Truth', color='blue')

        # Plot Predicted Trajectory
        ax.plot(pred_sample[:, 0].cpu().numpy(),
                pred_sample[:, 1].cpu().numpy(),
                pred_sample[:, 2].cpu().numpy(),
                label='Predicted', color='red', linestyle='dashed')

        # Labels and Title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'Trajectory Comparison - Sample {sample_num} - Epoch {epoch}')
        ax.legend()

        # Save the plot with epoch number at the start
        filename = os.path.join(destination_dir,
                                f'epoch_{epoch}_sample_{i}_action_mse.png')
        plt.savefig(filename)
        plt.close()


def alpha_bar(time_step):
    return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2


def calculate_betas(num_diffusion_timesteps, max_beta=0.999):
    """
    Calculate the betas (noise schedule) using the provided alpha_bar function.

    Args:
        num_diffusion_timesteps (int): The total number of diffusion timesteps.
        max_beta (float): The maximum allowed beta value.

    Returns:
        np.ndarray: The array of beta values for each timestep.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


def plot_avg_mse_with_betas(
    epoch: int,
    mse_per_timestep: dict,
    num_diffusion_timesteps: int,
    output_dir: str,
    max_beta: float = 0.999,
):
    """
    Plots the average MSE for each timestep and compares it to the noise schedule (betas).

    Args:
        mse_per_timestep (dict): A dictionary where the key is the timestep and the value is a list of MSEs at that timestep.
        num_diffusion_timesteps (int): The total number of diffusion timesteps.
        max_beta (float): The maximum allowed beta value for the noise schedule.
    """
    timesteps = []
    avg_mse = []

    # Ensure the 'plots' directory exists
    destination_dir = os.path.join(output_dir, 'plots')
    os.makedirs(destination_dir, exist_ok=True)

    # Calculate the average MSE for each timestep
    for t in sorted(mse_per_timestep.keys()):
        if len(mse_per_timestep[t]) > 0:
            timesteps.append(t)
            avg = (sum(mse_per_timestep[t])) / len(mse_per_timestep[t])
            avg_mse.append(avg)

    # Calculate betas (noise schedule)
    betas = calculate_betas(num_diffusion_timesteps, max_beta)

    # Plotting
    plt.figure(figsize=(12, 6))  # Increase width to 16 inches

    # Plot MSE without markers
    plt.plot(timesteps, avg_mse, linestyle='-', color='b', label='Average MSE', linewidth=2)

    # Plot noise schedule (betas) without markers
    plt.plot(range(num_diffusion_timesteps), betas, linestyle='--', color='r', label='Beta (Noise Schedule)', linewidth=2)

    # Adding titles and labels
    plt.title('Average MSE per Timestep vs. Noise Schedule (Betas)', fontsize=16)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Values', fontsize=14)

    # Set x-axis limits to make it start at the corner
    plt.xlim(left=timesteps[0], right=timesteps[-1])  # Ensure it starts at the first timestep

    # Set x-axis ticks to display every 10 timesteps
    plt.xticks(range(timesteps[0], timesteps[-1] + 1, 10))  # Ticks every 10 timesteps

    # Customize grid for better clarity
    plt.grid(True)

    # Display the legend
    plt.legend(loc='upper right')

    # Save the plot with epoch number at the start
    filename = os.path.join(destination_dir, f'epoch_{epoch}_timestep_performance.png')
    plt.savefig(filename)
    plt.close()