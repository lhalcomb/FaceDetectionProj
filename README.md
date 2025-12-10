# Face Clustering with FaceNet + HDBSCAN

An automated pipeline for detecting, embedding, and clustering faces in images using deep learning. The system leverages a pretrained FaceNet model for face embeddings and HDBSCAN for robust identity clustering, generating comprehensive artifacts for analysis and visualization.

## Features

- **Face Detection & Alignment**: MTCNN-based detection with automated alignment
- **Deep Face Embeddings**: InceptionResnetV1 (FaceNet) pretrained model
- **Intelligent Clustering**: HDBSCAN algorithm for density-based grouping
- **Rich Visualizations**: 3D PCA projections and dendrogram plots
- **Organized Outputs**: Per-cluster directories, CSV reports, and thumbnail grids

## Repository Structure

```
.
├── requirements.txt          # Python dependencies
├── test.py                   # Main pipeline script
├── notebooks/
│   └── analysis.ipynb       # Exploratory analysis notebook
└── artifacts/               # Generated outputs (created on run)
```

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: An alternative approach using `face_recognition` was explored. Ask about this in follow-up interview if you're interested in comparing methods.

## Usage

### Option A: Run the Python Script

```bash
python test.py
```

This executes the complete pipeline from face detection through clustering and visualization.

### Option B: Run the Jupyter Notebook

```bash
jupyter notebook
```
Also, to see the thumbnails.png you can just run the thumbCluster.py file. 
```bash
python thumbCluster.py
```

Navigate to `notebooks/analysis.ipynb` and run all cells for an interactive, step-by-step walkthrough.

## Output Structure

All results are saved to the `artifacts/` directory:

```
artifacts/
├── clusters.csv              # Cluster assignments: image_filename, cluster_id
├── outliers.csv             # List of outlier images (cluster_id = -1)
├── viz_3d.png               # 3D PCA visualization colored by cluster
├── Condensed_Tree.png       # HDBSCAN dendrogram 
├── cluster_0/               # Images belonging to cluster 0
│   ├── image_019.jpg
│   ├── image_139.jpg
|   |__ ...

│   └── thumbnail_grid.png   # Visual summary of cluster
├── cluster_1/
│   └── ...
└── outliers/                # Unclustered faces
    └── ...
```

### Output Files Explained

- **`clusters.csv`**: Two-column CSV mapping each image filename to its cluster ID (`-1` indicates outliers)
- **`outliers.csv`**: Dedicated list of images that couldn't be reliably clustered
- **`viz_3d.png`**: 3D scatter plot showing embedding space with color-coded clusters
- **`cluster_N/`**: Directories containing all faces assigned to cluster N, plus a thumbnail grid for quick review seen as **`thumbnail.png`**

## Pipeline Overview

The clustering pipeline follows these steps:

1. **Face Detection**: MTCNN detects and crops faces from input images
2. **Face Alignment**: Geometric normalization for consistent embedding quality
3. **Embedding Generation**: InceptionResnetV1 produces 512-dimensional face vectors
4. **Normalization**: L2 normalization of embeddings for cosine similarity
5. **Clustering**: HDBSCAN groups similar faces without requiring cluster count specification
6. **Visualization**: PCA reduction to 3D for interpretable plotting
7. **Export**: Organized outputs with CSV reports and per-cluster image directories

## Technical Details

- **Face Detector**: Multi-task Cascaded Convolutional Networks (MTCNN)
- **Embedding Model**: InceptionResnetV1 pretrained on VGGFace2
- **Clustering Algorithm**: Hierarchical DBSCAN (density-based, handles noise)
- **Dimensionality Reduction**: PCA for visualization only (clustering uses full embeddings)

## Notes

- Outliers (cluster `--1`) represent faces that don't fit well into any cluster—these may be low-quality detections, unique individuals with single photos, or significantly different poses/expressions
- HDBSCAN automatically determines the number of clusters based on data density
- The pipeline is fully reproducible given the same input images and random seed

---

**Questions?** Feel free to reach out or review the exploratory notebook for detailed methodology and parameter tuning insights. Hope to hear from you soon!