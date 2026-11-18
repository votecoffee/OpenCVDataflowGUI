# Sources and reproducibility

All images, edit masks, models and projects in this folder are original repository
controls under the root [MIT license](../../../LICENSE). They contain no customer
captures or third-party images. Every saved project has an on-canvas credit note.

`build_example.py` constructs 32×32 grayscale ring-like parts. Training uses nine
combinations of -1/0/+1 pixel horizontal/vertical offsets with independent seeded
noise (100–108). Held-out known views use seeds 21/22/23 with gains 1/0.93/1.07;
their pixels and hashes differ from training. Dark crack, missing arc and subtle
intensity edits are made only after holding out the query. No held-out defect
image contributes to the covariance or PCA fits.

[truth.json](truth.json) records exact asset SHA-256 values and construction
metadata. Edit masks record changed query pixels; the outlier block supplies a
window-level decision and no defect localization. The subtle edit is a known
accepted defect, retained to expose the selected descriptor/threshold limit.

[benchmark.py](benchmark.py) independently extracts native HOG, computes NumPy
sample covariance and regularized quadratic-form distances, and compares native
inversion/Mahalanobis results. It imports no application implementation. Its
recorded [benchmark](benchmark.json) includes raw and L2 modes and unsupported
zero-norm L2 input. Two saved PCA models and a matching covariance illustrate
revision compatibility, rather than establish a second defect-detection result.

The limit of 4 and Ridge choices are controlled demonstrations. Applications need
their own separate training, threshold tuning and validation sets.
