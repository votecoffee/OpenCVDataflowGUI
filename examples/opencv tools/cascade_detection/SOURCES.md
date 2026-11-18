# Sources and rights

| Files | Source and terms | Changes |
| --- | --- | --- |
| `assets/portrait.png` | NASA photograph of Eileen Collins, public domain via [scikit-image astronaut](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) | Copied unchanged from this repository's attributed object-landmark example; 512×512 pixels |
| `assets/haar.xml` | [OpenCV 4.13.0 frontal-face default](https://raw.githubusercontent.com/opencv/opencv/4.13.0/data/haarcascades/haarcascade_frontalface_default.xml), model credited to Rainer Lienhart | Unchanged bytes, including original Intel copyright/license/disclaimer |
| `assets/lbp.xml` | [OpenCV 4.13.0 LBP frontal face](https://raw.githubusercontent.com/opencv/opencv/4.13.0/data/lbpcascades/lbpcascade_frontalface.xml) under [upstream Apache-2 terms](https://raw.githubusercontent.com/opencv/opencv/4.13.0/LICENSE) | Unchanged bytes; original license copied to `LICENSE.opencv` |
| `assets/two-faces.png` | Derived from the same public-domain portrait | Original full-size view plus half-size INTER_AREA copy translated by [512,100] on a constant canvas |
| `assets/portrait16.png` | Same public-domain portrait | uint16 pixels equal original uint8×257 |
| `assets/blank.png`, graphs, scripts and coarse annotations | Original repository work, [MIT](../../../LICENSE) | Blank control, native configurations, overlay chains and approximate face rectangle |

`truth.json` records exact SHA-256 for all six image/model assets. The unchanged
upstream `LICENSE.opencv` SHA-256 is `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
The HAAR model's own Intel notice remains intact inside its XML and takes
precedence over the general OpenCV license for that file. Model files are not
relicensed as MIT. No downloaded Python code, retraining, customer images or
private examples were used.

Native behavior was checked against [OpenCV contrib 5.0.0 cascade implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/xobjdetect/src/cascadedetect.cpp)
and the [CascadeClassifier API](https://docs.opencv.org/4.13.0/d1/de5/classcv_1_1CascadeClassifier.html).
The 4.13 model bytes are explicitly qualified on the installed OpenCV 5 CPU path.
The approximate manual portrait ROI is not an independent accuracy dataset;
constructed second-view coordinates follow its known scale/translation.
