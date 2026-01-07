import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models import get_4channel_resnet
from torch.utils.data import DataLoader
from data_utils import SatellitePatchDataset
from tqdm import tqdm
from sklearn.cluster import KMeans

def run_test(patch_dir, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Model
    model = get_4channel_resnet(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    # 2. Get patch filenames to determine grid size
    patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith('.npy')])
    coords = [f.replace('patch_', '').replace('.npy', '').split('_') for f in patch_files]
    rows = [int(c[0]) for c in coords]
    cols = [int(c[1]) for c in coords]
    grid_h, grid_w = max(rows) + 1, max(cols) + 1

    # 3. Extract Features
    dataset = SatellitePatchDataset(patch_dir=patch_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    features = []
    
    print(f"🛰️ Scanning Sahara location ({grid_h}x{grid_w} grid)...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing Patches"):
            feat = model(batch.to(device))
            features.append(feat.cpu().numpy())
    
    feats = np.vstack(features)

    # 4. Clean Features (Fixes the NaN Error)
    # Replaces NaNs with 0.0 and infinity with large finite numbers
    if np.isnan(feats).any():
        print("⚠️ NaNs detected in features. Cleaning data...")
        feats = np.nan_to_num(feats, nan=0.0)

    # 5. Cluster
    print(f"🧩 Grouping into 4 terrain types...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(feats)

    # 6. Reshape and Visualize
    terrain_map = labels.reshape(grid_h, grid_w)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(terrain_map, cmap='tab10')
    plt.colorbar(label="Terrain ID")
    plt.title("Unsupervised Mapping: Sahara Desert Analysis")
    plt.savefig("sahara_test_result.png")
    plt.show()

if __name__ == "__main__":
    run_test("location_patches", "final_backbone.pth")