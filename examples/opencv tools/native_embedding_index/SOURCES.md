# Sources and attribution

The nine image files are copied unchanged from the repository's original
[K-nearest controls](<../knn_classification/README.md>). Those controls use original
synthetic stripe images from [reusable clusters](<../reusable_clusters/README.md>)
and one original diagonal image. Images, descriptors, native index artifacts,
projects, builder and tests are original OpenCVDataflowGUI work under the
repository's [MIT license](../../../LICENSE). No external/customer images,
learned weights or private-repository material are included.

All nine images and six native model bundles are SHA-256 pinned in
[truth.json](truth.json). Each project contains an attribution comment. HOG/native
models include the training descriptors and stable source filenames. Queries
use separate held-out pixels.

Native API references:

- [OpenCV ANNIndex documentation](https://docs.opencv.org/5.0/main_modules/features_annoy.html)
  describes create/add/build, query, native save/load and on-disk build.
- [OpenCV 5.0.0 wrapper source](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/features/src/annoy.cpp)
  establishes the float/uchar distance layouts, native output copying and seed
  conversion. The wrapper implementations here are original.
- [Bundled native index implementation](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/features/3rdparty/annoy/annoylib.h)
  describes metric and mmap behavior. OpenCV incorporates Annoy under its upstream
  Apache-2.0 terms. No implementation source is copied into these examples.

Qualification is on Python 3.14.3/OpenCV 5.0.0 contrib 5.0.0.93. The direct native
byte-Hamming binding returned zeros for distinct byte vectors on this Windows
build; the implemented fallback indexes unpacked bits with native Euclidean
distance, using `L2² = popcount(XOR)`. Original 256-bit controls verify distances
0/128/256. Direct native Hamming remains unqualified.

The native loader can report more roots than requested for tiny galleries; the
bundle records the measured post-reload count and validates it again on load.
Approximate candidate recall and real inspection accuracy need representative
validation. Checksums are integrity checks, not source authentication.
