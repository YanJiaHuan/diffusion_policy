import zarr
import os
import numpy as np
import matplotlib.pyplot as plt

def explore_zarr_structure(zarr_path):
    def print_group(group, indent=0):
        for key in group:
            print('  ' * indent + f"Key: {key}")
            item = group[key]
            if isinstance(item, zarr.hierarchy.Group):
                print('  ' * indent + f"Group: {key}/")
                print_group(item, indent + 1)
            else:
                print('  ' * indent + f"Dataset: {key} | Shape: {item.shape} | Dtype: {item.dtype}")
    
    def visualize_dataset(name, dataset):
        data = dataset[:]
        if data.ndim == 1:
            plt.figure()
            plt.plot(data)
            plt.title(f"{name} - Line Plot")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.show()
        elif data.ndim == 2:
            plt.figure()
            plt.imshow(data, aspect='auto')
            plt.title(f"{name} - Heatmap")
            plt.colorbar()
            plt.show()
        elif data.ndim == 3 and data.shape[-1] in [1, 3]:
            plt.figure()
            if data.shape[-1] == 1:
                plt.imshow(data.squeeze(), cmap='gray')
            else:
                plt.imshow(data.astype(np.uint8))
            plt.title(f"{name} - Image")
            plt.axis('off')
            plt.show()
        else:
            print(f"Visualization not supported for {name} with shape {dataset.shape}")

    # Open the Zarr store
    zarr_store = zarr.open(zarr_path, mode='r')
    
    print(f"Exploring Zarr dataset at: {zarr_path}")
    print("-" * 40)
    print_group(zarr_store)
    print("-" * 40)
    
    # Show basic statistics and visualize data if possible
    for group_name in zarr_store:
        group = zarr_store[group_name]
        for dataset_name in group:
            dataset = group[dataset_name]
            print(f"Dataset: {group_name}/{dataset_name}")
            print(f"  Shape: {dataset.shape}")
            print(f"  Dtype: {dataset.dtype}")
            print(f"  First few entries: {dataset[:5]}")
            print("-" * 40)
            
            # Try to visualize the dataset if it's numeric
            if np.issubdtype(dataset.dtype, np.number):
                visualize_dataset(f"{group_name}/{dataset_name}", dataset)

# Example usage
explore_zarr_structure("/home/zcai/jh_workspace/diffusion_policy/data/our_collected_data/test/replay_buffer.zarr")
