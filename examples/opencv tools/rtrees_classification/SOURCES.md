# Sources and license

All images, graphs and generator code are original work by OpenCVDataflowGUI
contributors, under the repository's [MIT license](../../../LICENSE).
No customer images, external photographs, pretrained weights or third-party
tutorial code/prose are included.

Nine original images are copied byte-for-byte from the
[K-nearest image study](../knn_classification/SOURCES.md). Geometric stripe
orientation supplies the two class labels independently of classification output.
The diagonal is an explicitly separate controlled image. All image pixel arrays
are distinct, including the training/query split.

Native OpenCV HOG and RTrees generate the feature vectors and frozen forests.
The builder documents the isolated fixture preparation; runtime training does
not set the caller's RNG seed. OpenCV remains covered by its upstream license.
Reference: [RTrees API](https://docs.opencv.org/4.x/d0/d65/classcv_1_1ml_1_1RTrees.html).
