# Sources and license

Images, models, graphs and authoring code are original work by OpenCVDataflowGUI
contributors under the repository's [MIT license](../../../LICENSE).
No external model, tutorial image, customer image or private plugin is included.

Nine synthetic stripe images are copied byte-for-byte from the repository's
[K-nearest example](../knn_classification/SOURCES.md). Six vertical/horizontal
training windows fit native HOG/PCA/K-nearest models. Two distinct held-out
vertical/horizontal windows and a separate diagonal control are excluded from
fitting. `truth.json` records every asset SHA-256, training order, labels, expected
project state and the deliberately selected residual threshold. All nine decoded
pixel hashes are distinct. This is controlled software evidence, not an independent
population sample or generalization study.

`benchmark.py` computes descriptors directly using the installed native
`cv2.HOGDescriptor`, then uses independent NumPy SVD, matrix projection and nearest
neighbors. It compares the saved PCA subspaces and raw/reduced classifications.
It does not import the app's PCA implementation or train on held-out images.
Shipped model JSON retains the mean/basis, source descriptor settings and model
revision; classifier JSON includes only the six projected training rows and labels.
