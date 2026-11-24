# Radon angular projection profiles

Open [parallel_stripes.json](parallel_stripes.json) and select **Run Once**.
The three parallel stripes produce peaks with a **38° line normal**, a
**52° line direction**, and signed distances of **0, −22, and +22 px**
from the reported origin. The bright vertical mark lies outside the
inspection rectangle. Select the Radon block's **Preview** output to see
the sinogram or **Image** to see the mapped lines.

Open [tilted_edge.json](tilted_edge.json) for a slanted intensity step.
Canny supplies its edge image to a descending 150°–90° sweep with 0.5°
spacing and center-circle cropping. The strongest peak has a **123°**
normal and **+9 px** signed distance. Both PNGs are original synthetic
fixtures covered by the repository license.

## Using the block

Wire a monochrome image or a color image to be interpreted as grayscale.
Use a source-grid mask or region to select contributing pixels. The circle
option intersects that selection with the native centered circular crop.
The reported source-grid line spans use the longest contiguous selected
portion for each peak.

**Projection** carries the native unnormalized sums as int32 for byte inputs
or float64 for higher depths. **Preview** contrast-stretches those sums to
uint8 for viewing. Index columns using **Axes → angle_degrees** and rows
using **distance_start_pixels + row × distance_step_pixels**. The distance
is relative to the reported source origin. For a normal angle θ, the
normal in source pixels is `(cos θ, −sin θ)`, with x right and y down.
The origin accounts for the exact native padding or circle crop.

Choose positive, negative, or absolute peak strength and adjust angular and
distance separation to distinguish nearby peaks. Set **Maximum peaks** to
zero for profiles alone. Angles follow OpenCV's rounded interval count.
Large searches can be split by cropping the input or reducing the number of
angles. [OpenCV Radon transform implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/radon_transform.cpp)
