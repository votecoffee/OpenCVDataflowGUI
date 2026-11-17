# Sources and license

Images, graphs and generator code are original work by OpenCVDataflowGUI
contributors, under the repository's [MIT license](../../../LICENSE).
No customer images, external photographs, pretrained weights, third-party tutorial
code or prose are included.

Six training images and two held-out images are copied byte-for-byte from the
original [reusable-cluster study](../reusable_clusters/SOURCES.md). The new diagonal
image is generated with native line drawing at the coordinates in the builder.
All nine image pixel arrays differ. Geometric stripe orientation determines the
two class labels independently of prediction results.

Native OpenCV HOG produces the training vectors. The JSON models contain those
vectors, their explicit orientation classes, normalization and native settings.
OpenCV remains covered by its upstream license; the [KNearest API](https://docs.opencv.org/4.x/dd/de1/classcv_1_1ml_1_1KNearest.html)
describes the native classifier.
