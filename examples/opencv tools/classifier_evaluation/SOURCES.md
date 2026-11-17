# Sources and licenses

- Creator: OpenCVDataflowGUI contributors. License: repository MIT license.
- Source: original stripe fixtures and native HOG/KNN model in
  [knn_classification](../knn_classification/SOURCES.md). Eight image files and
  `classifier-none.json` are copied unchanged, with SHA-256 checked against that
  example's `truth.json`. No third-party images, model weights or tutorial code.
- `mislabelled.json`: generated here by native OpenCV KNN using the same six HOG
  training rows with intentionally swapped class labels. This is an incorrect
  training control; the held-out image truth remains unchanged.
- Graphs, comments, grouping manifests and build code: original repository work.
- [truth.json](truth.json) records exact asset hashes. Models contain their
  original synthetic training feature rows and class mapping.

The confusion/metric definitions are standard formulas. Documentation links their
primary reference pages; no scikit-learn code or dependency is included.
