# Terrain Change Detection System

AI-powered unsupervised terrain change detection system using satellite imagery and deep learning.

## Features

- **Independent Clustering Analysis**: Separate AI analysis for Before/After images to preserve terrain characteristics
- **4-Channel Satellite Support**: Processes RGB + NIR satellite imagery
- **Interactive Streamlit Dashboard**: User-friendly web interface for analysis
- **Comprehensive Visualizations**:
  - Before/After terrain maps
  - Change detection heatmap
  - Terrain distribution pie charts
  - Net change analysis
  - Transition matrix
  - High-resolution change maps

## Tech Stack

- **Deep Learning**: ResNet50 backbone with SimCLR
- **Clustering**: KMeans for unsupervised terrain classification
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Geospatial**: Rasterio for TIF file processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aiml-project.git
cd aiml-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your model weights:
- Download `final_backbone.pth` (trained ResNet50 model)
- Place it in the project root directory

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

### Workflow

1. **Tab 1 - Data Preparation**
   - Input paths to Before.tif and After.tif satellite images
   - Click "Process Both Images" to extract patches

2. **Tab 2 - AI Analysis**
   - Click "Run Change Detection Analysis"
   - View Before, After, and Change Heatmap visualizations

3. **Tab 3 - Change Analytics**
   - Explore detailed statistics and visualizations
   - View terrain distributions, transitions, and changes

## Project Structure

```
├── app.py                      # Main Streamlit application
├── models.py                   # ResNet50 4-channel model definition
├── requirements.txt            # Python dependencies
├── final_backbone.pth          # Pre-trained model weights
├── data_utils.py              # Data processing utilities
├── train.py                   # Model training script
└── README.md                  # This file
```

## Terrain Types

The system detects 8 terrain classes:

- 🔴 Urban/Built (Red - #e74c3c)
- 🟢 Vegetation (Green - #27ae60)
- 🟠 Barren/Soil (Orange - #f39c12)
- 🔵 Water Bodies (Blue - #3498db)
- 🟣 Agriculture (Purple - #9b59b6)
- ⚫ Rocky/Mountain (Gray - #95a5a6)
- 🟢 Mixed Forest (Teal - #16a085)
- ⚫ Industrial (Dark - #34495e)

## How It Works

1. **Patch Extraction**: Satellite images are divided into 64×64 patches
2. **Feature Extraction**: ResNet50 extracts deep features from each patch
3. **Independent Clustering**: KMeans clusters each image separately to preserve terrain characteristics
4. **Change Detection**: Compares cluster assignments to identify changed areas
5. **Visualization**: Generates comprehensive analytics and visualizations

## Key Design Decision: Independent Clustering

The system uses **independent clustering** for Before and After images rather than unified clustering. This ensures:
- Natural terrain characteristics are preserved for each location
- Different geographical regions (e.g., Bengaluru vs Tokyo) are analyzed correctly
- Water bodies, urban areas, and vegetation are accurately classified

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- Satellite imagery in 4-channel TIF format (RGB + NIR)

## License

MIT License

## Acknowledgments

- ResNet50 architecture from torchvision
- SimCLR framework from lightly
- Satellite imagery processing with rasterio
