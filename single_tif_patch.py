import os
import rasterio
import numpy as np
from tqdm import tqdm

# ==========================================
# 📂 FOLDER & PATH CONFIGURATION
# ==========================================
INPUT_TIF = "D:/ssl_training_data/all_training_data/ssl_training_mosaic_desert_sahara.tif"  # Change this to your .tif location
OUTPUT_DIR = "location_patches"              # Where the patches will be saved
PATCH_SIZE = 64

def create_patches(tif_path, save_dir, size=64):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with rasterio.open(tif_path) as src:
        # Get dimensions
        width = src.width
        height = src.height
        
        # Calculate how many patches fit
        n_w = width // size
        n_h = height // size
        
        print(f"🛰️ Processing: {tif_path}")
        print(f"📐 Grid Size: {n_h} rows x {n_w} columns ({n_h * n_w} total patches)")

        for i in tqdm(range(n_h), desc="Patching Rows"):
            for j in range(n_w):
                # Define the window to crop
                window = rasterio.windows.Window(
                    j * size, i * size, size, size
                )
                
                # Read 4-channel data
                data = src.read(window=window)
                
                # Ensure patch is the correct size (ignores edges)
                if data.shape[1] == size and data.shape[2] == size:
                    patch_name = f"patch_{i}_{j}.npy"
                    np.save(os.path.join(save_dir, patch_name), data)

if __name__ == "__main__":
    create_patches(INPUT_TIF, OUTPUT_DIR, PATCH_SIZE)
    print(f"✅ Patching complete! Files saved in: {OUTPUT_DIR}")