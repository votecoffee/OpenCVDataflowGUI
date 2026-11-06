# Adaptive background correction

Open [adaptive_background.json](adaptive_background.json) with
**File → Open…** and click **Run once**. The image has a left-to-right
lighting gradient and a dark circular defect. On the raw image, a
fixed threshold marks 2,400 background pixels and misses the defect.
After correction, the same threshold isolates all 81 defect pixels.

The block is under **Image processing → Color and depth**. Use
**Closing** to estimate the background around small dark details,
**Opening** for small bright details, or **Gaussian** for a smooth
local average. Set a radius larger than the details to preserve.
**Background** exposes the full estimate in raw sample units.
[OpenCV morphology reference](https://docs.opencv.org/5.0/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html)

**Subtract** handles additive variation. **Divide** handles
multiplicative shading and reports near-zero denominators through
**Invalid mask**. Those pixels retain their original values. A mask
or region limits corrected writes. The estimator uses neighborhoods
from the full image. Color channels are corrected independently,
and alpha is preserved.

**Target level** sets the desired baseline. For integer images,
it is a fraction of full scale. Floating-point images use raw
sample units. Select floating output to keep signed differences.

```powershell
opencv-dataflow-run examples/image_processing/adaptive_background/adaptive_background.json
```

[All examples](../../README.md)
