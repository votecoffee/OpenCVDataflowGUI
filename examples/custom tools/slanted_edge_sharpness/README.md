# Slanted-edge sharpness

Open `project.json`, then **Run all**. The sharp image passes and the blurred
image fails the minimum MTF50 limit of **0.15 cycles/pixel**. Both images are
original 16-bit synthetic fixtures under this repository's MIT license.

The measurement fits one straight, 5-degree slanted edge, bins its samples
into a subpixel edge-spread function (ESF), differentiates it, and measures
the windowed spatial-frequency response (SFR). Open its result tab for the
ESF/SFR chart and numeric report. The image result has the fitted edge and
measurement label as drawable graphics.

The underlying Gaussian blur has sigma **0.8** or **2.0 pixels**. Before window
effects, its analytic MTF50 is approximately **0.2342** or **0.09370 cycles/pixel**,
and its 10–90% edge width is **2.5631 × sigma**. The default Hann window makes
the measured MTF50 slightly higher; keep the window and sampling width fixed
when comparing images. A larger band reduces window bias for broader blur.
These fixtures sample the analytic Gaussian step at pixel centers; they do not
model sensor pixel integration, sharpening, noise or a camera's tone curve.

Select a single near-vertical or near-horizontal edge with both plateaus when
using real images. A mask or region can define the measurement area. Use linear
luminance, avoid saturation, and set sample pitch and length unit explicitly
for physical frequencies. Frequencies use the current image's pixel grid;
resized images require the corresponding pitch. A crossing outside the native
Nyquist band is reported as a bound, with no invented scalar result.

This is an engineering estimate, **not an ISO 12233 compliance measurement**.
The implementation uses the ESF → derivative → windowed DFT approach and
central-difference response correction described by Burns et al.,
[Updated Camera Spatial Frequency Response for ISO 12233 (2022)](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/ei/34/9/IQSP-357).
The algorithm and fixtures here were implemented independently; no third-party
code or image assets are included.

Regenerate with `python "examples/custom tools/slanted_edge_sharpness/generate.py"`
from the repository root.
