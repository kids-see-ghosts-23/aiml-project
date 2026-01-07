import numpy as np
import matplotlib.pyplot as plt
import os

def inspect_clusters(sample_dir="cluster_samples"):
    groups = sorted(os.listdir(sample_dir))
    
    for group in groups:
        group_path = os.path.join(sample_dir, group)
        patches = [f for f in os.listdir(group_path) if f.endswith('.npy')]
        
        fig, axes = plt.subplots(1, len(patches), figsize=(15, 5))
        fig.suptitle(f"Inspection: {group}", fontsize=16)
        
        for i, patch_name in enumerate(patches):
            data = np.load(os.path.join(group_path, patch_name))
            
            # Convert 4-channel satellite data to viewable RGB
            # We take the first 3 channels and normalize for display
            rgb = data[:3, :, :].transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            
            axes[i].imshow(rgb)
            axes[i].set_title(patch_name)
            axes[i].axis('off')
            
        plt.show()

if __name__ == "__main__":
    inspect_clusters()