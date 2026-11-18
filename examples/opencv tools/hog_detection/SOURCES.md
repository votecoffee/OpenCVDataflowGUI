# Sources and rights

The 19 PNG images, their known boxes, example graphs and custom linear/RBF SVM/HOG
models were generated for this repository by `build_example.py`. They contain no
customer images, outside captures or derived third-party training data. They are
covered by the repository [MIT license](../../../LICENSE). The query and training
windows use different noise seeds. `truth.json` records SHA-256 for every image and
model, including the intentionally corrupted model used for rejection testing.

Default and Daimler people coefficients are supplied by the installed OpenCV
runtime and are not copied into this folder. Their blank-image projects check
native API integration only. The implementation uses OpenCV 5's HOGDescriptor,
setSVMDetector and detectMultiScale APIs, now exposed through xobjdetect in the
contrib source tree. Native implementation reference:
[OpenCV contrib HOG](https://github.com/opencv/opencv_contrib/blob/5.x/modules/xobjdetect/src/hog.cpp).

The independent benchmark uses native one-window descriptors and independently
enumerates locations/scales, calculates weighted margins and suppresses boxes.
This checks detector integration against separate calculations, without claiming
an independent implementation of HOG feature extraction or an accuracy benchmark.
