# Face Embedding Visualization Project

This project uses a pre-trained FaceNet model to generate face embeddings and visualize them on facial images.

## � Quick Setup

### 1. Download the Model Files
The FaceNet model files are stored separately due to size limitations:

**�📁 Download from Google Drive:**
[FaceNet Model Files](https://drive.google.com/drive/folders/17-MR7fSc342OcIneN0bV6mKX_j02jska?usp=sharing)

**📋 Setup Instructions:**
1. Download the model files from the Google Drive link above
2. Create a `models/` folder in your project directory
3. Extract/place the model files in the `models/` folder
4. Your structure should look like:
   ```
   models/
   ├── 20180402-114759/
   │   ├── 20180402-114759.pb
   │   ├── model-20180402-114759.ckpt-275.data-00000-of-00001
   │   ├── model-20180402-114759.ckpt-275.index
   │   └── model-20180402-114759.meta
   └── 20180402-114759.zip (optional)
   ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
## 📁 Project Structure

```
EZ pic/
├── models/                          # Pre-trained models (download separately)
│   └── 20180402-114759/            # FaceNet model directory (from Google Drive)
│       ├── 20180402-114759.pb      # Frozen TensorFlow model
│       ├── model-20180402-114759.ckpt-275.data-00000-of-00001
│       ├── model-20180402-114759.ckpt-275.index
│       └── model-20180402-114759.meta
├── images/                          # Input images for processing
│   ├── sample.png                  # Main test image
│   └── unnamed.png                 # Additional image
├── scripts/                         # Python scripts
│   ├── org_trial_script.py         # Main embedding generation script
│   └── embedding_visualizer.py     # Visualization script
├── outputs/                         # Generated visualizations and results
│   ├── embedding_visualization.png # Comprehensive visualization
│   └── embedding_analysis.png      # Detailed analysis charts
├── trial_script.py                 # Main runner script with menu
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## ⚠️ Important: Model Files Required
**Before running the scripts, you MUST download the model files from:**
[Google Drive - FaceNet Models](https://drive.google.com/drive/folders/17-MR7fSc342OcIneN0bV6mKX_j02jska?usp=sharing)

## 🚀 Usage

**⚠️ Prerequisites: Make sure you've downloaded the model files from Google Drive first!**

### Quick Start (Recommended)
```bash
python trial_script.py
```
This provides an interactive menu with options:
1. **Basic embedding generation** (org_trial_script.py)
2. **Embedding + visualization** (embedding_visualizer.py) 
3. **🚀 HYBRID pipeline** (ezpic_hybrid_pipeline.py) - **RECOMMENDED**
4. **Sequential execution** (both scripts)

### 🚀 **NEW: EzPic Hybrid Strategy**
The hybrid pipeline implements the optimal face processing strategy:

| Step | Method | Purpose | Performance |
|------|--------|---------|-------------|
| 1. Detection | **BlazeFace/MediaPipe** | Fast face location + landmarks | ~10-20ms |
| 2. Alignment | **MTCNN Math** (CPU only) | Perfect face alignment | ~1-2ms |
| 3. Embedding | **FaceNet** | 512D face vector | ~30-50ms |
| **Total** | **Hybrid Pipeline** | **Complete processing** | **~50-70ms** ✅ |

**Target: Sub-100ms processing for real-time applications**

### Manual Usage
```bash
cd scripts
python org_trial_script.py              # Basic: Pre-processed images → FaceNet
python embedding_visualizer.py          # Visualization of embeddings
python ezpic_hybrid_pipeline.py         # HYBRID: Full detection + alignment + embedding
```

### 📊 **Processing Approaches Comparison**

| Approach | Speed | Accuracy | Use Case |
|----------|--------|----------|-----------|
| **Original** (org_trial_script.py) | Fast | High* | Pre-cropped faces |
| **Hybrid** (ezpic_hybrid_pipeline.py) | **Fastest** | **Highest** | **Real-time applications** |
| **Visualization** (embedding_visualizer.py) | Medium | High | Analysis & debugging |

*Requires manual face cropping and alignment

### What the scripts do:
- **org_trial_script.py**: Loads the FaceNet model and generates 512D face embeddings
- **embedding_visualizer.py**: Creates visual mappings of embeddings onto facial features
- **trial_script.py**: Interactive menu runner for easy execution
```
This script:
- Runs the embedding generation
- Creates visual mappings of embeddings onto the face
- Saves comprehensive visualizations to `../outputs/`

## 📊 Output Visualizations

### embedding_visualization.png
A 6-panel comprehensive view showing:
- **Original Image**: Your input face image
- **Heatmap Overlay**: Which facial regions contribute most to the embedding
- **Pure Heatmap**: Raw embedding intensity map
- **Feature Points**: Key embedding dimensions marked on the face
- **Statistics**: Embedding metrics and properties
- **Dimension Plot**: First 100 embedding dimensions

### embedding_analysis.png
Detailed analysis charts:
- **Full Vector Plot**: Complete 512D embedding visualization
- **Value Distribution**: Histogram of embedding values
- **Top Features**: 20 most significant embedding dimensions

## 🎯 Understanding the Results

- **Heatmap Colors**: 
  - Red/Yellow = High embedding values (strong facial features)
  - Blue/Purple = Low embedding values (less significant areas)
- **Feature Points**: Numbered circles show the most important embedding dimensions
- **Embedding Vector**: 512 floating-point numbers that uniquely represent the face

## 🔧 Requirements

- Python 3.x
- TensorFlow 2.x (with v1 compatibility)
- OpenCV (`cv2`)
- PIL/Pillow
- NumPy
- Matplotlib

## 📝 Technical Details

- **Model**: Pre-trained FaceNet (20180402-114759)
- **Input Size**: 160x160 pixels
- **Output**: 512-dimensional embedding vector
- **Preprocessing**: Image resizing and pre-whitening normalization

## 🎨 Customization

To use your own images:
1. Place new images in the `images/` folder
2. Update `IMAGE_PATH` in `scripts/trial_script.py`
3. Run the scripts to generate new visualizations

## 🔬 Research Applications

This visualization can help understand:
- Which facial features are most important for face recognition
- How different faces map to embedding space
- The distribution and patterns of facial embeddings
- Feature importance for identity verification

---

*Generated on October 3, 2025*