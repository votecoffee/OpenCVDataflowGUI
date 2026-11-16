# Restore blurred frames from neighbors

Open a project below and press **Run once**. Select **Restore detail from
neighbors** to inspect its restored image, original target, restored mask,
donor-index preview and per-neighbor report. No downloads or plugins are needed.
The ordered captures use matching pixel indices; **pixel_grid** explicitly
asserts this generated registration.

| Project | Expected result |
| --- | --- |
| [project.json](project.json) | Sharp / horizontally blurred / sharp captures; restore the middle target. Restored samples match independent sharp truth exactly. |
| [moving_parts.json](moving_parts.json) | Target part at x100, donor part at x40. Preserve both changed areas in the target while restoring static detail elsewhere. |
| [highdepth.json](highdepth.json) | Preserve 16-bit BGR values with per-capture coverage masks and a target inspection region. |
| [gaussian.json](gaussian.json) | Restore a target blurred with the configured 7x7, sigma-1.5 Gaussian. |
| [unchanged.json](unchanged.json) | Equally sharp captures: pass through unchanged with an explicit `unchanged` report. |
| [no_neighbor_fail.json](no_neighbor_fail.json) | Deliberately fail because no sharper neighbor exists. The original image still emits. |

The verification blocks compare restored pixels to a separately generated sharp
image, with zero allowable sample error. The before block measures error at the
same pixels before restoration. Inspect the full source image as well: pixels
outside the restored mask are copied exactly and are not included in that
zero-error assertion. The target capture keeps its ticket; the two donors are
reported independently. Passing restoration is not a part judgment.

Choose the expected target blur, error tolerance and change halo for your data.
The mismatch gate does not prove that small or low-contrast defects are absent.
Keep raw target images for critical defect inspection. See the
[full contracts and limitations](../../../docs/neighbor-restoration.md).

All eight images, projects and the [generator](build_example.py) are original
OpenCVDataflowGUI fixtures under the repository's [MIT license](../../../LICENSE).
No customer images or third-party images are included. An on-canvas comment in
every project records this attribution. Regenerate with the application Python:

```powershell
python "examples/opencv tools/neighbor_restoration/build_example.py"
```

The [truth manifest](truth.json) records known blur/part locations and asset
SHA256 hashes. Automated tests check exact regeneration and both Debug/Lean
outcomes. OpenCV's [filter reference](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html)
describes the underlying filters; the unavailable native video deblurrer is not
required.
