# Local-contrast edges under uneven lighting

Open [compare_uneven_lighting.json](compare_uneven_lighting.json) and select
**Run Once**. Two faint plate outlines vary by about five gray levels over a
sloping background with a broad highlight and shadow. The graph compares each
edge map with the known boundary pixels inside a shared inspection rectangle.
Select either comparison block to see matched, unexpected and missed edges.

At a **1.5 px** localization tolerance, **Local-contrast edges** produced two
connected outlines with **100% precision and recall** on this fixture. The
ordinary Canny branch uses the same fine smoothing and comparable thresholds.
It had **15.4% precision and 99.7% recall** at the same tolerance. These scores
describe the saved synthetic scene. The source and truth PNGs are original
fixtures covered by the repository license.

## Using the block

Wire a monochrome or color image. **Short range** sets the fine Gaussian
window. **Long range** estimates broader lighting with a box mean or Gaussian
window. Choose a long window wider than the edge transition. **Contrast**
outputs the signed difference between the fine image and the background.
**Strength** shows the gradient magnitude, and **Image** contains the
binary Canny edges selected by the two hysteresis thresholds.

Thresholds use 8-bit-equivalent intensity units per pixel. Float intensities
from 0 to 1 and uint16 full-scale intensities map to the byte range before
filtering. Smoothing and hysteresis use the full source before a mask or
region restricts the three image outputs. The block accepts up to 4,194,304
pixels per image.

The supported Python wheel lacks the native BrightEdges binding. This block
uses OpenCV filters and derivative Canny to cover the local-contrast
workflow. [OpenCV BrightEdges implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/brightedges.cpp)
