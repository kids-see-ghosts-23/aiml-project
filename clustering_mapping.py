import torch
import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from models import get_4channel_resnet
from data_utils import SatellitePatchDataset
from tqdm import tqdm

def extract_features(patch_dir, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the "Brain"
    backbone = get_4channel_resnet(pretrained=False)
    backbone.load_state_dict(torch.load(weights_path, map_location=device))
    backbone.to(device).eval()

    # 2. Load Patches (No transforms needed for this)
    dataset = SatellitePatchDataset(patch_dir=patch_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    features = []
    print(f"🧠 Extracting terrain features from {len(dataset)} patches...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing Batches"):
            batch = batch.to(device)
            feat = backbone(batch)
            features.append(feat.cpu().numpy())
    
    return np.vstack(features)

def map_terrains(features, n_clusters=5):
    print(f"🧩 Grouping into {n_clusters} terrain types...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

def identify_clusters(patch_dir, labels, output_dir="cluster_samples", num_samples=5):
    # Get list of all patch filenames
    patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith('.npy')])
    
    # Create a folder for each cluster
    for i in range(len(np.unique(labels))):
        cluster_folder = os.path.join(output_dir, f"terrain_group_{i}")
        os.makedirs(cluster_folder, exist_ok=True)
        
        # Find all patches belonging to this cluster
        indices = np.where(labels == i)[0]
        sample_indices = random.sample(list(indices), min(num_samples, len(indices)))
        
        print(f"📁 Saving samples for Group {i}...")
        for idx in sample_indices:
            src = os.path.join(patch_dir, patch_files[idx])
            dst = os.path.join(cluster_folder, patch_files[idx])
            shutil.copy(src, dst)

if __name__ == "__main__":
    PATCH_DIR = "./train_patches"  # Your folder of .npy patches
    WEIGHTS = "final_backbone.pth"
    N_CLUSTERS = 4

    # 1. Check if we already saved the features to avoid the 1-hour wait
    if os.path.exists("extracted_features.npy"):
        print("📂 Loading previously extracted features...")
        feats = np.load("extracted_features.npy")
    else:
        # Run the heavy extraction only if necessary
        feats = extract_features(PATCH_DIR, WEIGHTS)
        np.save("extracted_features.npy", feats) # Save for next time!

    # 2. Run the clustering
    terrain_labels = map_terrains(feats, n_clusters=N_CLUSTERS)
    np.save("terrain_labels.npy", terrain_labels) # Save your labels!

    # 3. Identify what each group is
    identify_clusters(PATCH_DIR, terrain_labels)

    # 4. Visualize
    plt.figure(figsize=(10, 5))
    plt.hist(terrain_labels, bins=range(N_CLUSTERS + 1), rwidth=0.8, color='skyblue')
    plt.title("Terrain Distribution Found by AI")
    plt.xlabel("Terrain Group ID")
    plt.ylabel("Number of Patches")
    plt.xticks(range(N_CLUSTERS))
    plt.show()
    
    print("✅ Analysis Complete! Check the 'cluster_samples' folder to see your results.")