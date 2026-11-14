# Coverage-mask resizing

Open a project with **File → Open**. The block is under **OpenCV → Image
Processing (imgproc) → Geometry → Resample coverage mask**.

| Project | Demonstrates |
| --- | --- |
| [Coverage thresholds](coverage_thresholds.json) | Downsample a mask containing borders, a hole, thin invalid lines and alternating columns. Compare full, half and any-positive area thresholds and accepted-pixel counts. |
| [Fractional chain](fractional_chain.json) | Carry coverage through aligned resize stages, compare with the original mask resampled directly, and show the information lost by thresholding early. |
| [Incompatible target](incompatible_target.json) | Deliberate contained error: a crop has the requested size but its coordinate transform does not match a whole-grid resize. No resampled mask is emitted. |

The original synthetic 256×192 mask was created by OpenCVDataflowGUI contributors
and is covered by this repository's MIT license. It contains no external or
customer images. Each project includes an on-canvas source/scope comment.

**Coverage** is a float64 fraction of each output pixel's source footprint. The
default threshold of one accepts only fully valid footprints. Threshold one-half
is inclusive; zero means any positive coverage and never admits empty footprints.
The heatmap is a diagnostic display; the binary Mask output connects to ordinary
mask inputs. Empty masks are successful processing results.

The target input supplies dimensions and checks known pixel frames. It does not
perform registration. Without a target, configure Width and Height directly.
Use **Input mode → Coverage** for fractional float32/64 inputs in [0,1]. Keep
fractional outputs through a resize chain when needed; nonaligned repeated grids
can still accumulate approximation, so direct resampling of the original mask is
preferred. These are pixel-cell area measurements, not interpolation-kernel
support certificates or physical calibration.

Run the two successful projects without the GUI:

```powershell
python -m opencv_dataflow.runner "examples/opencv tools/mask_resampling/coverage_thresholds.json"
python -m opencv_dataflow.runner "examples/opencv tools/mask_resampling/fractional_chain.json"
```

The incompatible-target project intentionally returns a nonzero runner status.
Regenerate the fixture and all three graphs without network access:

```powershell
python "examples/opencv tools/mask_resampling/build_example.py"
```

See [mask contracts](../../../docs/regions.md#resample-validity-mask-coverage).
