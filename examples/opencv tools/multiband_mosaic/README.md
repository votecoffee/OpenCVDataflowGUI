# Multiband inspection mosaics

Seven projects reuse a real VisA PCB photograph with known crops and intensity
gains. They demonstrate the **Multiband** option on Stitch image pair, Stitch scan
sequence and Render mosaic tile under **OpenCV → Image Stitching → Geometry**.
No plugin or optional dependency is needed.

| Project | Run | Expected |
| --- | --- | --- |
| [Pair, Float64](pair_float64.json) | Run once | Two uint16 crops, exact known 110-pixel translation |
| [Pair, native detail](pair_native_detail.json) | Run once | OpenCV CPU int16 blender, uint8 inputs and float weights |
| [Pair, native int16 weights](pair_native_int16_weights.json) | Run once | Alternative native weight arithmetic |
| [Scan, Float64](scan_float64.json) | Run all | Register three uint16 crops and blend overlaps |
| [Scan with exposure gain](scan_exposure_gain.json) | Run all | Fit gain to the anchor, then compose three uint8 views |
| [Tiled Float64](tiles_float64.json) | Run all | Register once and render windows 0, 5 and 11 with aligned halos |
| [Native depth guard](native_depth_guard.json) | Run once | Deliberate validation failure; choose Float64 to accept uint16 |

Select the mosaic/tile block and inspect Image, Coverage, Overlap and Layout/Tile.
The report records backend, requested/effective bands, workspace estimate and
source transforms/tickets. Alter **Multiband bands** to compare transition widths;
zero averages overlap without reductions. Original defaults remain priority
composition, so existing projects retain their behavior.

The default Float64 backend retains uint16 and floating-point precision until the
final output cast. The native detail backend requires uint8 gray/RGB/BGR and
contains integer rounding, even with float weight maps. It runs on CPU and can
differ visibly from Float64. Neither method repairs misregistration or moving
objects. Blended overlap is modified image data; use coverage and the original
source views for measurements that require unmodified samples.

All three crops come from one image. Apparent illumination changes were authored
with gains 1, 0.85 and 0.70. The uint16 images are eight-bit derivatives expanded
by 257, not new 16-bit sensor measurements. Estimated scan transforms can add
small margins to the nominal 480×320 canvas. [truth.json](truth.json) contains the
nominal geometry, formulas and file hashes; [SOURCES.md](SOURCES.md) provides
**CC BY 4.0** image attribution, repeated on every project canvas. Example code
and graphs are MIT.

From the repository root:

```powershell
python -m opencv_dataflow.runner "examples/opencv tools/multiband_mosaic/scan_exposure_gain.json" --all-images
python "examples/opencv tools/multiband_mosaic/build_example.py"
```

The builder deterministically regenerates assets/projects from the already
bundled photograph. See [multiband contracts](../../../docs/multiband-mosaics.md)
for masks, alpha, memory limits, tiling and native precision.
