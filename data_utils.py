import os
import glob
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2


# 1. Patching Logic (The Cookie Cutter)
def create_patches(input_dir, output_dir, patch_size=64):
    """
    Finds all .tif files and slices them into small squares.
    """
    os.makedirs(output_dir, exist_ok=True)
    tif_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".tif")
    ]

    patch_count = 0
    print(
        f"🪚 Slicing {len(tif_files)} large images into {patch_size}x{patch_size} patches..."
    )

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            # Step through the image in increments of 'patch_size'
            for x in range(0, src.width - patch_size, patch_size):
                for y in range(0, src.height - patch_size, patch_size):
                    window = rasterio.windows.Window(x, y, patch_size, patch_size)
                    patch = src.read(window=window)

                    # Data Cleaning: Ignore patches with NaN or that are completely empty/black
                    if not np.isnan(patch).any() and np.any(patch > 0):
                        filename = f"patch_{patch_count}.npy"
                        np.save(os.path.join(output_dir, filename), patch)
                        patch_count += 1

    print(f"✅ Created {patch_count} patches in: {output_dir}")


# 2. SimCLR Augmentation Logic (The Hall of Mirrors)
class SatelliteTransform:
    def __init__(self, input_size=64):
        # We use geometric transforms that work on 4 channels simultaneously
        self.transform = v2.Compose(
            [
                v2.RandomResizedCrop(size=input_size, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=90),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

    def __call__(self, x):
        # Returns TWO different versions of the SAME image
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


# 3. Dataset Class (The Librarian)
class SatellitePatchDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        self.patch_files = glob.glob(os.path.join(patch_dir, "*.npy"))
        self.transform = transform
        print(f"📂 Dataset ready with {len(self.patch_files)} patches found.")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        # Load the raw math data (.npy)
        patch = np.load(self.patch_files[idx])
        # Convert to a PyTorch Tensor [Channels, Height, Width]
        patch_tensor = torch.from_numpy(patch).float()

        if self.transform:
            return self.transform(patch_tensor)
        return patch_tensor
