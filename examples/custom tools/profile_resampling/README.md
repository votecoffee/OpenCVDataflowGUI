# Common-pitch image profiles

Open a project and choose **Run once**. Each uses the supplied original 16-bit
image; no plugin or download is required. Select the resamplers to compare their
plots and inspect Profile/Report results.

| Project | Demonstrates |
| --- | --- |
| [Common pixel grid](pixel_common.json) | Two horizontal paths sampled at 1 and 0.5 pixel spacing share an exact 4-pixel target grid, retaining their own image coordinates. |
| [Common physical grid](physical_common.json) | A declared 0.25 mm/px rectified workplane converts both profiles to a common 1 mm pitch. |
| [Masked decimation](masked_decimation.json) | Compare antialiased filtering with unfiltered sampling. A taught gap stays invalid and expands with the filter footprint. |

The image contains a known slow cosine, a high-frequency sine and a Gaussian
peak centered at x=320. Filtering suppresses the high-frequency alias while
retaining the peak's location. The two path rows have identical intensities;
their source points remain at y=16 and y=48 even when distance grids agree.
In the physical example, x=320 corresponds to path distance 80 mm.

In the masked example, columns 220..239 are excluded. Null results mean missing
support, not zero intensity. Kaiser support makes a wider invalid neighborhood
than unfiltered sampling. Reflection at the outer endpoints is declared in the
report's `boundary_extended` flags; choose Invalid boundary for strict edge
support. Exact pitch omits a nonaligned final endpoint rather than stretching it.

The original MIT image, calibration fixture, generator and projects are
OpenCVDataflowGUI contributor work, with on-canvas attribution. No customer or
third-party images are included. `truth.json` records the image formula, peak,
scale, mask and SHA256 hashes. The calibration is synthetic, without camera or
gauge acceptance. This example is under Custom tools because the supported
Float64 resampler is an original implementation; the installed OpenCV signal
resampler did not qualify the tested coordinate/antialiasing requirements.

Automated checks verify common grids, source pixels/tickets, exact regeneration,
peak location, gap support and 0.05% intensity agreement between different source
sampling densities. The masked filtered residual against the known low-frequency
signal is below 3 raw units RMS over tested interior support; unfiltered RMS
exceeds 1000 raw units. Unit tests separately check analytic amplitude and phase.

Regenerate with the application environment:

```powershell
.venv/Scripts/python.exe "examples/custom tools/profile_resampling/build_example.py"
```

See [contracts and native audit](../../../docs/profile-resampling.md) and
[all examples](../../README.md).
