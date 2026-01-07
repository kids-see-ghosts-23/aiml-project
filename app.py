import os
os.environ["TORCH_CLASSES_SKIP_PATH_EXAMINATION"] = "1"
import torch
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from models import get_4channel_resnet
from sklearn.cluster import KMeans
import seaborn as sns

st.set_page_config(page_title="Terrain Change Detection", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Metric Card Styling */
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;}
.metric-card h2 {margin: 0; font-size: 2.5em;}
.metric-card p {margin: 5px 0 0 0; opacity: 0.9;}
.change-metric {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}

/* Fix for Tab Visibility */
.stTabs [data-baseweb="tab-list"] {gap: 12px;}
.stTabs [data-baseweb="tab"] {
    background-color: #e0e4ef !important; 
    border-radius: 8px 8px 0 0; 
    padding: 12px 24px;
    color: #1a1c24 !important; /* Forces dark text for unselected tabs */
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #667eea !important; 
    color: white !important; /* Forces white text for the active tab */
}
""", unsafe_allow_html=True)

st.title("Terrain Change Detection System")
st.markdown("*AI-powered analysis of satellite imagery for urban and environmental monitoring*")

WEIGHTS = "final_backbone.pth"
PATCH_SIZE = 64

TERRAIN_NAMES = {
    0: "Urban/Built",
    1: "Vegetation", 
    2: "Barren/Soil",
    3: "Water Bodies",
    4: "Agriculture",
    5: "Rocky/Mountain",
    6: "Mixed Forest",
    7: "Industrial"
}

TERRAIN_COLORS = ['#e74c3c', '#27ae60', '#f39c12', '#3498db', '#9b59b6', '#95a5a6', '#16a085', '#34495e']

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_4channel_resnet(pretrained=False)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device,weights_only=True))
    model.to(device).eval()
    return model, device

st.sidebar.header("Configuration")
n_clusters = st.sidebar.slider("Terrain Classes", 2, 8, 4, help="Number of distinct terrain types to detect")
st.sidebar.markdown("---")
st.sidebar.markdown("### Detected Terrains")
for i in range(n_clusters):
    color = TERRAIN_COLORS[i]
    terrain_name = TERRAIN_NAMES.get(i, f'Terrain {i}')
    st.sidebar.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 20px; height: 20px; background-color: {color}; border-radius: 3px; margin-right: 10px;"></div><strong>{terrain_name}</strong></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Data Preparation", "AI Analysis", "Change Analytics"])

with tab1:
    st.header("Upload Satellite Images")
    st.markdown("Provide paths to **Before** and **After** satellite images (.tif format)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before Image")
        before_path = st.text_input("Path to Before.tif:", r"C:\Users\Karan\Downloads\archive\ssl_training_data-20251101T222621Z-1-002\ssl_training_data\ssl_training_mosaic_urban_bengaluru.tif", key="before")
    with col2:
        st.subheader("After Image") 
        after_path = st.text_input("Path to After.tif:", r"C:\Users\Karan\Downloads\archive\ssl_training_data-20251101T222621Z-1-002\ssl_training_data\ssl_training_mosaic_urban_dense_tokyo.tif", key="after")
    
    st.markdown("---")
    if st.button("Process Both Images", use_container_width=True):
        if os.path.exists(before_path) and os.path.exists(after_path):
            progress = st.progress(0)
            status_text = st.empty()
            
            for idx, (label, path, out_dir) in enumerate([("Before", before_path, "patches_before"), ("After", after_path, "patches_after")]):
                status_text.text(f"Processing {label} image...")
                os.makedirs(out_dir, exist_ok=True)
                with rasterio.open(path) as src:
                    n_w, n_h = src.width // PATCH_SIZE, src.height // PATCH_SIZE
                    for i in range(n_h):
                        for j in range(n_w):
                            window = rasterio.windows.Window(j*PATCH_SIZE, i*PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
                            data = src.read(window=window)
                            if not np.isnan(data).any() and np.any(data > 0):
                                np.save(os.path.join(out_dir, f"patch_{i}_{j}.npy"), data)
                progress.progress((idx + 1) / 2)
            
            status_text.empty()
            progress.empty()
            st.success("Both images processed successfully!")
        else:
            st.error("One or both file paths are invalid")

with tab2:
    st.header("AI Terrain Analysis")
    st.markdown("Independent clustering analyzes each image separately to preserve terrain characteristics")
    
    if st.button("Run Change Detection Analysis", use_container_width=True):
        before_dir, after_dir = "patches_before", "patches_after"
        
        if os.path.exists(before_dir) and os.path.exists(after_dir):
            with st.spinner("Running AI analysis..."):
                model, device = load_model(WEIGHTS)
                st.info(f"Computing on: {device}")
                
                def extract_features(patch_dir, label):
                    patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith('.npy')])
                    coords = [f.replace('patch_', '').replace('.npy', '').split('_') for f in patch_files]
                    grid_h, grid_w = max(int(c[0]) for c in coords) + 1, max(int(c[1]) for c in coords) + 1
                    
                    features = []
                    progress = st.progress(0, text=f"Extracting {label} features...")
                    with torch.no_grad():
                        for idx, f_name in enumerate(patch_files):
                            data = np.load(os.path.join(patch_dir, f_name))
                            patch_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)
                            feat = model(patch_tensor)
                            features.append(feat.cpu().numpy())
                            progress.progress((idx + 1) / len(patch_files), text=f"Extracting {label} features... {idx+1}/{len(patch_files)}")
                    progress.empty()
                    return np.vstack(features), patch_files, grid_h, grid_w
                
                feats_before, files_before, h_before, w_before = extract_features(before_dir, "Before")
                feats_after, files_after, h_after, w_after = extract_features(after_dir, "After")
                
                st.success("Features extracted!")
                
                feats_before = np.nan_to_num(feats_before, nan=0.0)
                feats_after = np.nan_to_num(feats_after, nan=0.0)
                
                with st.spinner("Clustering terrains..."):
                    kmeans_before = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels_before = kmeans_before.fit_predict(feats_before)
                    
                    kmeans_after = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels_after = kmeans_after.fit_predict(feats_after)
                
                def build_map(labels, files, h, w):
                    terrain_map = np.full((h, w), -1, dtype=float)
                    for f_name, label in zip(files, labels):
                        parts = f_name.replace('patch_', '').replace('.npy', '').split('_')
                        r, c = int(parts[0]), int(parts[1])
                        terrain_map[r, c] = label
                    return terrain_map
                
                map_before = build_map(labels_before, files_before, h_before, w_before)
                map_after = build_map(labels_after, files_after, h_after, w_after)
                
                st.session_state['map_before'] = map_before
                st.session_state['map_after'] = map_after
                st.session_state['labels_before'] = labels_before
                st.session_state['labels_after'] = labels_after
                st.session_state['n_clusters'] = n_clusters
                
                st.success("Analysis complete!")
            
            st.markdown("""<style>
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;}
.metric-card h2 {margin: 0; font-size: 2.5em;}
.metric-card p {margin: 5px 0 0 0; opacity: 0.9;}
.change-metric {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}

/* Fix for the Tabs */
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background-color: #f0f2f6; 
    border-radius: 8px 8px 0 0; 
    padding: 10px 20px;
    color: #31333F; /* Ensures text is dark and visible on the light gray background */
}
.stTabs [aria-selected="true"] {
    background-color: #667eea !important; 
    color: white !important; /* Forces text to be white when tab is active */
}
</style>
""", unsafe_allow_html=True)
            st.subheader("Terrain Maps Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(TERRAIN_COLORS[:n_clusters])
            
            with col1:
                st.markdown("#### Before")
                fig1, ax1 = plt.subplots(figsize=(8, 8))
                ax1.imshow(map_before, cmap=cmap, vmin=0, vmax=n_clusters-1)
                ax1.axis('off')
                ax1.set_title('Before Analysis', fontsize=16, fontweight='bold', pad=20)
                st.pyplot(fig1)
            
            with col2:
                st.markdown("#### After")
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                ax2.imshow(map_after, cmap=cmap, vmin=0, vmax=n_clusters-1)
                ax2.axis('off')
                ax2.set_title('After Analysis', fontsize=16, fontweight='bold', pad=20)
                st.pyplot(fig2)
            
            with col3:
                st.markdown("#### Change Heatmap")
                change_map = (map_before != map_after).astype(float)
                change_map[map_before == -1] = np.nan
                change_map[map_after == -1] = np.nan
                
                fig3, ax3 = plt.subplots(figsize=(14, 10))
                im = ax3.imshow(change_map, cmap='hot', interpolation='bilinear', aspect='auto')
                ax3.set_title('High-Resolution Change Detection Map', fontsize=16, fontweight='bold', pad=20)
                ax3.set_xlabel('Patch Column', fontsize=12)
                ax3.set_ylabel('Patch Row', fontsize=12)
                cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                cbar.set_label('Change Detected', rotation=270, labelpad=20, fontsize=11, weight='bold')
                plt.tight_layout()
                st.pyplot(fig3)
                
                st.session_state['change_map'] = change_map
        else:
            st.error("Process images in Tab 1 first!")

with tab3:
    st.header("Change Analytics Dashboard")
    
    if 'map_before' in st.session_state and 'map_after' in st.session_state:
        map_before = st.session_state['map_before']
        map_after = st.session_state['map_after']
        labels_before = st.session_state['labels_before']
        labels_after = st.session_state['labels_after']
        n_clusters = st.session_state['n_clusters']
        change_map = st.session_state['change_map']
        
        unique_before, counts_before = np.unique(labels_before, return_counts=True)
        unique_after, counts_after = np.unique(labels_after, return_counts=True)
        pct_before = (counts_before / len(labels_before)) * 100
        pct_after = (counts_after / len(labels_after)) * 100
        
        total_pixels = np.sum(~np.isnan(change_map))
        changed_pixels = np.nansum(change_map)
        change_percentage = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        st.markdown("### Overall Change Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card change-metric">
                <h2>{change_percentage:.1f}%</h2>
                <p>Total Area Changed</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{int(changed_pixels):,}</h2>
                <p>Patches Modified</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{n_clusters}</h2>
                <p>Terrain Classes</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Terrain Distribution Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Before Distribution")
            for i, pct in enumerate(pct_before):
                color = TERRAIN_COLORS[i]
                terrain_name = TERRAIN_NAMES.get(i, f"Terrain {i}")
                st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 10px;"><div style="width: 15px; height: 15px; background-color: {color}; border-radius: 2px; margin-right: 10px;"></div><span style="font-size: 18px; font-weight: 600;">{pct:.2f}%</span><span style="margin-left: 10px; color: #666;">{terrain_name}</span></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### After Distribution")
            for i, pct in enumerate(pct_after):
                color = TERRAIN_COLORS[i]
                terrain_name = TERRAIN_NAMES.get(i, f"Terrain {i}")
                delta = pct - pct_before[i] if i < len(pct_before) else pct
                delta_color = '#27ae60' if delta > 0 else '#e74c3c' if delta < 0 else '#95a5a6'
                st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 10px;"><div style="width: 15px; height: 15px; background-color: {color}; border-radius: 2px; margin-right: 10px;"></div><span style="font-size: 18px; font-weight: 600;">{pct:.2f}%</span><span style="margin-left: 10px; color: {delta_color}; font-weight: bold;">({delta:+.2f}%)</span><span style="margin-left: 10px; color: #666;">{terrain_name}</span></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Visual Analytics")
        
        pct_before_full = np.zeros(n_clusters)
        pct_after_full = np.zeros(n_clusters)
        for i, pct in zip(unique_before, pct_before):
            pct_before_full[i] = pct
        for i, pct in zip(unique_after, pct_after):
            pct_after_full[i] = pct
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        terrain_colors = TERRAIN_COLORS[:n_clusters]
        labels = [TERRAIN_NAMES.get(i, f"Terrain {i}") for i in range(n_clusters)]
        
        wedges1, texts1, autotexts1 = axes[0].pie(pct_before_full, labels=labels, colors=terrain_colors, autopct='%1.1f%%',
                                                   startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
        axes[0].set_title('Before - Terrain Distribution', fontsize=14, fontweight='bold', pad=20)
        
        wedges2, texts2, autotexts2 = axes[1].pie(pct_after_full, labels=labels, colors=terrain_colors, autopct='%1.1f%%',
                                                   startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
        axes[1].set_title('After - Terrain Distribution', fontsize=14, fontweight='bold', pad=20)
        
        for autotext in autotexts1 + autotexts2:
            autotext.set_color('white')
            autotext.set_fontsize(9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### Net Change Analysis")
        
        fig2, ax = plt.subplots(figsize=(14, 6))
        change_delta = pct_after_full - pct_before_full
        x = np.arange(n_clusters)
        colors_bar = [TERRAIN_COLORS[i] if change_delta[i] >= 0 else '#e74c3c' for i in range(n_clusters)]
        
        bars = ax.bar(x, change_delta, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax.set_ylabel('Change (%)', fontsize=13, fontweight='bold')
        ax.set_title('Terrain Change (After - Before)', fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([TERRAIN_NAMES.get(i, f"T{i}") for i in range(n_clusters)], rotation=45, ha='right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, val) in enumerate(zip(bars, change_delta)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.markdown("---")
        st.markdown("### Transition Matrix")
        st.markdown("*Note: Since clustering is independent, cluster IDs may not align perfectly. Use with caution.*")
        
        transition_matrix = np.zeros((n_clusters, n_clusters))
        valid_mask_before = (map_before >= 0) & (map_before < n_clusters)
        valid_mask_after = (map_after >= 0) & (map_after < n_clusters)
        valid_mask = valid_mask_before & valid_mask_after
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                transition_matrix[i, j] = np.sum((map_before[valid_mask] == i) & (map_after[valid_mask] == j))
        
        transition_pct = (transition_matrix / transition_matrix.sum()) * 100
        
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        im = ax3.imshow(transition_pct, cmap='YlOrRd', aspect='auto')
        
        ax3.set_xticks(np.arange(n_clusters))
        ax3.set_yticks(np.arange(n_clusters))
        ax3.set_xticklabels([TERRAIN_NAMES.get(i, f"T{i}") for i in range(n_clusters)], rotation=45, ha='right')
        ax3.set_yticklabels([TERRAIN_NAMES.get(i, f"T{i}") for i in range(n_clusters)])
        
        ax3.set_xlabel('After (To)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Before (From)', fontsize=12, fontweight='bold')
        ax3.set_title('Terrain Transition Matrix (%)', fontsize=14, fontweight='bold', pad=15)
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                text = ax3.text(j, i, f'{transition_pct[i, j]:.1f}',
                               ha="center", va="center", color="white" if transition_pct[i, j] > transition_pct.max()/2 else "black",
                               fontsize=10, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Percentage (%)', rotation=270, labelpad=20, fontsize=11)
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.markdown("---")
        st.markdown("### Detailed Change Heatmap")
        
        fig4, ax4 = plt.subplots(figsize=(14, 10))
        im4 = ax4.imshow(change_map, cmap='hot', interpolation='bilinear', aspect='auto')
        ax4.set_title('High-Resolution Change Detection Map', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Patch Column', fontsize=12)
        ax4.set_ylabel('Patch Row', fontsize=12)
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('Change Detected', rotation=270, labelpad=20, fontsize=11, weight='bold')
        plt.tight_layout()
        st.pyplot(fig4)
        
        st.markdown("---")
        st.markdown("### Terrain Naming Logic")
        st.info("""
        **How terrain names are assigned:**
        
        The AI performs **independent clustering** on each image separately. This preserves the natural terrain characteristics of each location.
        
        **Key points:**
        
        1. **Independent Analysis**: Before and After images are clustered separately, so terrain IDs maintain consistent meaning within each image
        2. **Spectral Characteristics**: NDVI, brightness, texture patterns in 4-channel satellite data (RGB + NIR)
        3. **Manual Mapping**: KMeans assigns arbitrary IDs (0, 1, 2...) - we manually map them to terrain labels via the `TERRAIN_NAMES` dictionary
        4. **Color Consistency**: The `TERRAIN_COLORS` array defines hex colors for each cluster ID across all visualizations
        
        **Why independent clustering?**
        - Different locations have different terrain compositions
        - Bengaluru (urban) vs Tokyo (coastal) have vastly different spectral signatures
        - Independent clustering ensures Water in Tokyo actually represents water, not misclassified as another terrain
        
        **To customize:**
        - Run analysis to see terrain maps
        - Identify which cluster ID represents which terrain
        - Edit both `TERRAIN_NAMES` and `TERRAIN_COLORS` dictionaries in app.py
        - Example: If cluster 1 is green and shows forests → set TERRAIN_NAMES[1] = "Vegetation"
        """)
    else:
        st.warning("Run AI Analysis in Tab 2 first to see statistics")
