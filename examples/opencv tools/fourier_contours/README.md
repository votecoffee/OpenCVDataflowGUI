# Fourier contour descriptors and alignment

Open [Align Fourier contours](align_fourier_contours.json) and choose
**Run Once**. The two contour detectors extract ordered outlines. The
spectrum blocks sample each closed perimeter 128 times, then express its
centered, unit-RMS coordinates as complex Fourier coefficients. The fit
uses 16 positive and 16 negative harmonics to estimate a similarity
transform. **Warp sample to reference** applies that matrix to the sample
image.

The original rasterized sample is rotated by 23° and scaled by 1.24×.
The saved fit recovers **−23.62°** and **0.8078×** in the sample-to-reference
direction. The aligned outline has a **1.33 px** RMS residual. Open
[Reject altered contour](reject_altered_contour.json) to see the deliberate
notch failure. That fit leaves a **6.08 px** RMS residual. Its preview keeps
the measured outline, and its transform output is withheld.

The fit accepts point lists, contour collections, or full descriptors at the
selected sample count. Compact descriptors keep the requested low
positive and negative frequencies and are available for downstream analysis.
The cyclic starting-point offset is part of the correspondence. The
spatial matrix contains rotation, scale and translation in image
coordinates. Rotation follows the image x and y axes, so positive angles
are clockwise on a standard image display. The residual gate uses RMS
error divided by the reference's RMS radius. A bounded spectral phase search
can recover minima missed by native root selection. The report identifies the
selected fit. Symmetry can produce more than one valid alignment.

The three silhouette PNGs are original MIT fixtures. Rebuild the projects
and assets with `python build_example.py --output <directory>`.
[OpenCV Fourier descriptor reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_fourier.html)
