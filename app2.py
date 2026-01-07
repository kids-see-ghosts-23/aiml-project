import os
os.environ["TORCH_CLASSES_SKIP_PATH_EXAMINATION"] = "1"
import torch
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
from models import get_4channel_resnet
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

# --- UI CONFIGURATION ---
st.set_page_config(page_title="High-Res SSL Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; height: 150px;}
.metric-card h2 {margin: 0; font-size: 2.5em;}
.change-metric {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
.stTabs [data-baseweb="tab"] { background-color: #e0e4ef !important; border-radius: 8px 8px 0 0; color: #1a1c24 !important; font-weight: 600;}
.stTabs [aria-selected="true"] { background-color: #667eea !important; color: white !important;}
.change-box {padding: 12px; margin: 8px 0; background-color: #1e1e1e; border-radius: 6px; border-left: 5px solid;}
</style>
""", unsafe_allow_html=True)

# --- BACKEND SETTINGS ---
WEIGHTS = "final_backbone.pth"
PATCH_SIZE = 64
TERRAIN_COLORS = ['#e74c3c', '#27ae60', '#f39c12', '#3498db', '#9b59b6', '#95a5a6', '#16a085', '#34495e']
TERRAIN_NAMES = {0: "Urban", 1: "Vegetation", 2: "Barren", 3: "Water", 4: "Agri", 5: "Rocky", 6: "Forest", 7: "Industrial"}

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_4channel_resnet(pretrained=False)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    model.to(device).eval()
    return model, device

# --- SIDEBAR ---
st.sidebar.header("Global Controls")
n_clusters_input = st.sidebar.slider("Number of Clusters", 2, 8, 4)
sensitivity = st.sidebar.slider("Heatmap Sensitivity", 0.0, 1.0, 0.2)
st.sidebar.markdown("---")
st.sidebar.markdown("### Detected Terrains Legend")

# Loop through the selected number of clusters to build the legend
for i in range(n_clusters_input):
    color = TERRAIN_COLORS[i]
    # Get the name from the dictionary, or use a default if not found
    terrain_name = TERRAIN_NAMES.get(i, f'Terrain {i}')
    
    # Render the color box and the name side-by-side using HTML
    st.sidebar.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="
                width: 20px; 
                height: 20px; 
                background-color: {color}; 
                border-radius: 3px; 
                margin-right: 10px;
                border: 1px solid #555;">
            </div>
            <strong style="color: white;">{terrain_name}</strong>
        </div>
        """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📁 Data Prep", "🧠 SSL AI Engine", "📊 Detailed Analytics"])

with tab1:
    st.subheader("Image Pre-processing")
    b_path = st.text_input("Before Path:", r"C:\Users\Karan\Downloads\archive\ssl_training_data-20251101T222621Z-1-002\ssl_training_data\ssl_training_mosaic_urban_bengaluru.tif")
    a_path = st.text_input("After Path:", r"C:\Users\Karan\Downloads\archive\ssl_training_data-20251101T222621Z-1-002\ssl_training_data\ssl_training_mosaic_urban_dense_tokyo.tif")
    
    if st.button("Generate Patches", use_container_width=True):
        for label, path, out_dir in [("Before", b_path, "patches_before"), ("After", a_path, "patches_after")]:
            os.makedirs(out_dir, exist_ok=True)
            with rasterio.open(path) as src:
                for i in range(src.height // PATCH_SIZE):
                    for j in range(src.width // PATCH_SIZE):
                        window = rasterio.windows.Window(j*PATCH_SIZE, i*PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
                        data = src.read(window=window)
                        if not np.isnan(data).any() and np.any(data > 0):
                            np.save(os.path.join(out_dir, f"patch_{i}_{j}.npy"), data)
        st.success("TIF Processing Complete.")

with tab2:
    st.header("🧠 SSL AI Terrain Analysis")
    st.markdown("Independent clustering is now replaced with **Joint Feature Clustering** to ensure terrain consistency.")
    
    if st.button("Run SSL Change Detection Analysis", use_container_width=True):
        before_dir, after_dir = "patches_before", "patches_after"
        
        if os.path.exists(before_dir) and os.path.exists(after_dir):
            with st.spinner("Initializing SSL Model..."):
                model, device = load_model(WEIGHTS)
                st.info(f"Computing on: {device}")
                
                def extract_features_with_progress(patch_dir, label):
                    patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith('.npy')])
                    features = []
                    
                    progress = st.progress(0, text=f"Extracting {label} features...")
                    with torch.no_grad():
                        for idx, f_name in enumerate(patch_files):
                            data = np.load(os.path.join(patch_dir, f_name))
                            # Normalize for SSL Weights (0-1 range)
                            patch_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device) / 255.0
                            feat = model(patch_tensor)
                            features.append(feat.cpu().numpy().flatten())
                            
                            if idx % 5 == 0:  # Update progress every 5 patches for performance
                                progress.progress((idx + 1) / len(patch_files), 
                                                 text=f"SSL Processing {label}: {idx+1}/{len(patch_files)}")
                    progress.empty()
                    return np.vstack(features), patch_files

                # 1. Feature Extraction
                feats_before, files_before = extract_features_with_progress(before_dir, "Before")
                feats_after, files_after = extract_features_with_progress(after_dir, "After")
                
                st.success("✅ SSL Features Extracted Successfully!")
                
                # 2. Joint Clustering (The AI logic that keeps labels the same)
                with st.spinner("Aligning Terrain Labels..."):
                    combined_feats = np.vstack([feats_before, feats_after])
                    # We use the n_clusters variable defined in the sidebar
                    kmeans = KMeans(n_clusters=n_clusters_input, random_state=42, n_init=10)
                    kmeans.fit(combined_feats)
                    
                    labels_before = kmeans.predict(feats_before)
                    labels_after = kmeans.predict(feats_after)

                # 3. SSL Change Magnitude (Cosine Distance)
                dist = 1 - F.cosine_similarity(torch.from_numpy(feats_before), torch.from_numpy(feats_after)).numpy()
                # Contrast Stretching for the heatmap
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)

                # 4. Map Reconstruction
                def build_map(vals, files):
                    coords = [f.replace('patch_', '').replace('.npy', '').split('_') for f in files]
                    h, w = max(int(c[0]) for c in coords) + 1, max(int(c[1]) for c in coords) + 1
                    grid = np.full((h, w), -1.0)
                    for v, c in zip(vals, coords):
                        grid[int(c[0]), int(c[1])] = v
                    return grid

                map_before = build_map(labels_before, files_before)
                map_after = build_map(labels_after, files_after)
                mag_map = build_map(dist, files_before)

                # Store in session state for Tab 3
                st.session_state.update({
                    'm_b': map_before, 
                    'm_a': map_after, 
                    'm_mag': mag_map, 
                    'l_b': labels_before, 
                    'l_a': labels_after,
                    'n_clusters': n_clusters_input
                })

                st.success("Analysis complete! Results ready in Analytics Tab.")

            # --- Visual Comparison ---
            st.subheader("Terrain Analysis Comparison")
            col1, col2, col3 = st.columns(3)
            cmap = ListedColormap(TERRAIN_COLORS[:n_clusters_input])
            
            with col1:
                st.markdown("#### Before Analysis")
                fig1, ax1 = plt.subplots(); ax1.imshow(map_before, cmap=cmap); ax1.axis('off'); st.pyplot(fig1)
            
            with col2:
                st.markdown("#### After Analysis")
                fig2, ax2 = plt.subplots(); ax2.imshow(map_after, cmap=cmap); ax2.axis('off'); st.pyplot(fig2)
            
            with col3:
                st.markdown("#### Feature Change Intensity")
                fig3, ax3 = plt.subplots(); im = ax3.imshow(mag_map, cmap='hot'); plt.colorbar(im); ax3.axis('off'); st.pyplot(fig3)
        else:
            st.error("Please process images in Tab 1 first!")
with tab3:
    if 'm_b' in st.session_state:
        # Fix for the KeyError: n_clusters is now safely retrieved
        n_cls = st.session_state.get('n_clusters', 4)
        m_mag = st.session_state['m_mag']
        
        # --- 1. METRICS ---
        total_p = np.sum(m_mag != -1)
        changed_p = np.sum(m_mag > sensitivity)
        ch_pct = (changed_p / total_p) * 100

        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f'<div class="metric-card change-metric"><h2>{ch_pct:.1f}%</h2><p>Area Changed</p></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><h2>{int(changed_p):,}</h2><p>Patches Changed</p></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><h2>{n_cls}</h2><p>Terrain Classes</p></div>', unsafe_allow_html=True)

        # --- 2. THE DETAILED HEATMAP (FIXED) ---
        st.markdown("### 🛰️ Ultra-High-Resolution Change Analysis")
        
        # Apply Power-Law (Gamma) to pop the hot spots
        gamma_mag = np.power(np.clip(m_mag, 0, 1), 1.6)
        # Apply slight Gaussian Blur to fix the "blocky" look
        smoothed = gaussian_filter(gamma_mag, sigma=0.4)
        masked_heat = np.ma.masked_where(m_mag == -1, smoothed)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(masked_heat, cmap='hot', interpolation='bicubic', aspect='auto')
        plt.colorbar(im, label="SSL Feature Shift")
        ax.set_title("Detection Map: Hot colors indicate significant structural change", fontsize=12)
        st.pyplot(fig)

        # --- 3. TRANSITION MATRIX ---
        st.markdown("---")
        st.markdown("### Terrain Transition Analysis")
        mat = np.zeros((n_cls, n_cls))
        for b, a in zip(st.session_state['l_b'].astype(int), st.session_state['l_a'].astype(int)):
            if b < n_cls and a < n_cls: mat[b, a] += 1
        
        fig_mat, ax_mat = plt.subplots(figsize=(10, 8))
        sns.heatmap(mat, annot=True, fmt='.0f', cmap='YlOrRd', 
                    xticklabels=[TERRAIN_NAMES.get(i, f"T{i}") for i in range(n_cls)],
                    yticklabels=[TERRAIN_NAMES.get(i, f"T{i}") for i in range(n_cls)])
        st.pyplot(fig_mat)

        # --- 4. SUMMARY CARDS ---
        st.markdown("### Change Summary")
        u_b, c_b = np.unique(st.session_state['l_b'], return_counts=True)
        u_a, c_a = np.unique(st.session_state['l_a'], return_counts=True)
        
        # Calculate Shift
        for i in range(n_cls):
            count_before = c_b[list(u_b).index(i)] if i in u_b else 0
            count_after = c_a[list(u_a).index(i)] if i in u_a else 0
            diff = (count_after - count_before) / total_p * 100
            
            if abs(diff) > 0.05:
                color = TERRAIN_COLORS[i]
                st.markdown(f"""
                <div class="change-box" style="border-left-color: {color};">
                    <strong style="color:{color};">{TERRAIN_NAMES.get(i, f"Terrain {i}")}</strong>: 
                    {'Increased' if diff > 0 else 'Decreased'} by <b>{abs(diff):.2f}%</b>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💡 Run the AI Analysis in Tab 2 to populate these results.")