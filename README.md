# GenLift Take-Home: Unsupervised Face Grouping

## Objective

Group unlabeled face photos by person and produce a machine-readable mapping. Your solution should be reproducible, explainable, and runnable on a typical laptop (CPU-only).

## What you’re given

- `faces/` — JPEG images named `image_001.jpg`, `image_002.jpg`, …
  Filenames are intentionally randomized; **do not assume any ordering leaks identity**.

We will validate your pipeline on a **private holdout set** during review.

## What to deliver (core)

1. **Embeddings → Clusters**

   - Extract face embeddings with an off-the-shelf backbone of your choice.
   - Cluster images by identity.

2. **Artifacts (place in `artifacts/`)**

   - `clusters.csv` — two columns: `image_filename,cluster_id` (use `-1` for outliers).
   - `outliers.csv` — list of filenames flagged as outliers (if any).
   - `viz_3d.png` — 3D projection colored by cluster.
   - _(Optional but appreciated)_ 2–3 thumbnail grids for representative clusters.

3. **Notebook + brief report**

   - A single **Jupyter notebook** (`notebooks/analysis.ipynb`) that:

     - runs end-to-end on the provided `faces/`,
     - explains parameter choices (e.g., distance metric, `eps`/`min_samples`),
     - shows diagnostics (k-distance plot or similar) and the 3D viz,
     - notes failure modes and what you’d improve with more time.

4. **Reproducibility**

   - `requirements.txt` or `pyproject.toml`
   - `run.sh` (or a clear README section) that executes your notebook non-interactively (e.g., `papermill`/`jupyter nbconvert`) and writes the artifacts above.
   - Set seeds where applicable.

## Constraints & guidelines

- **Runtime/Hardware:** CPU-only; aim for ≤30 minutes on a modest laptop, ≤4GB RAM.
- **Allowed libs:** numpy, pandas, scikit-learn, pillow/opencv, torch/tf (pretrained backbones), faiss (optional), umap-learn, matplotlib.
- **Evaluation hint:** For self-checking, compute a simple cluster quality proxy (e.g., pairwise precision/recall on a few self-labeled pairs you inspect). We will use our own checks on the holdout.

## Stretch (pick 1–2 if time allows)

- Simple CLI (`cluster --images faces/ --out artifacts/`).
- Small parameter sweep with a table/plot of quality vs. `eps`/`min_samples`.
- “Data-retention” exercise: emit an audit manifest showing how you would delete raw images while retaining embeddings.

## Submission

- Share a Git repo or zipped folder with:

  - `faces/` **excluded**, but your code assumes that structure.
  - `notebooks/analysis.ipynb`, `artifacts/` (from a sample run), environment files, and `run.sh`.

- Include a short top-level `README.md` with exact run commands.

## Review

We will:

1. Recreate your environment, run your pipeline on our holdout, and confirm artifacts.
2. Skim your notebook for reasoning grounded in your own outputs.
3. Do a brief live follow-up (e.g., adjust a parameter or swap distance metric) to understand your approach.

---

### Attribution

Portions of this dataset are derived from the **Caltech Faces Dataset (1999)**. The data are used here solely for educational and evaluation purposes. Caltech is not affiliated with this exercise.
