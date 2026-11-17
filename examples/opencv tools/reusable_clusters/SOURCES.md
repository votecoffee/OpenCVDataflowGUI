# Sources and license

All images, graphs and generator code in this folder are original work by
OpenCVDataflowGUI contributors and use the repository's [MIT license](../../../LICENSE).
No external photographs, customer images, tutorial text, pretrained weights or
third-party sample code are included.

The color images are mathematical three-panel controls; the white defect rectangle
is independently specified in `truth.json`. Stripe images use native OpenCV line
drawing at disjoint training/query positions. The native OpenCV HOG descriptor
and K-means algorithm generate the small JSON vocabulary and palette assets.
Those assets contain only numerical centers and declared preprocessing settings.

OpenCV's implementation remains governed by its own upstream license. API reference:
[OpenCV clustering](https://docs.opencv.org/4.x/d5/d38/group__core__cluster.html).
