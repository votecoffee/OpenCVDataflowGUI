# Reuse a calibrated mosaic layout

Open a project below with **File → Open…**, then choose **Run all**. The source
emits its two or three image files in the listed order; the batch waits until the
set is complete. `single_view.json` needs one acquisition. The final Render mosaic
tile node contains Image, Coverage, Overlap and Tile metadata. The projects do not
write files or connect to hardware.

Three original grayscale images have known offsets 0, 144 and 288 on a 480×128
canvas. Small deliberate brightness differences make source priority visible.
The sealed layout in `assets/calibration.json` pins the known geometry rather
than fitting an answer from the test image. A file SHA256 also pins its exact
bytes. All frozen replay projects perform no feature registration.

| Project | Expected final result |
| --- | --- |
| [full.json](full.json) | Pass; all three views in calibration order. |
| [reordered.json](reordered.json) | Pass; right/left/middle arrive out of order, with exactly the same output pixels and blend priority as Full. |
| [missing_middle.json](missing_middle.json) | Pass with incomplete status; right/left leave the middle's unique area uncovered. |
| [missing_anchor.json](missing_anchor.json) | Deliberate fail with partial data and `mosaic_sources_missing`; right/middle keep the same full canvas origin even without Left. |
| [single_view.json](single_view.json) | Pass with incomplete status; Right occupies its original position on the full canvas. |
| [missing_error.json](missing_error.json) | Deliberate validation error; missing Middle emits no render output. |
| [changed_grid.json](changed_grid.json) | Deliberate validation error; a one-pixel change to declared source coordinates invalidates strict reuse, even though dimensions match. |
| [multiband_tile.json](multiband_tile.json) | Pass; reordered images and circular source-mask holes, float64 multiband composition, one 64×64 tile at canvas origin (192, 0). |
| [native_registration.json](native_registration.json) | Pass; fit a new stable-ID layout from the three overlapping views, then render it. Fitted geometry is separate from the frozen known-geometry projects. |

Tile's `source_matching` lists calibrated/present/missing IDs and grid evidence.
Contributor records retain calibrated indices, current input indices and filename
tickets. The image payload has a stable `mosaic:<layout_id>` root; tile-local
coordinates map back to the unchanged full canvas. Missing captures are never
filled with images from an earlier run. A passing partial result asserts the
configured policy, rather than full coverage.

To use a newly fitted layout later, add **Save mosaic layout**, connect Layout
from Register scan layout and choose a new destination `.json`. The default write
mode preserves existing files. In a later project, **Load mosaic layout** feeds
Render mosaic tile; provide stable camera IDs through group labels, per-image
payload metadata or an explicit list/input. Existing ordered layouts remain
strict about input order. See the [full contract](../../../docs/mosaic-layout-reuse.md)
for grid validation, persistence, ticket association and resource limits.

The image fixtures, layout, generator and projects are original work by
OpenCVDataflowGUI contributors under the repository's [MIT license](../../../LICENSE).
They contain no customer or third-party images. On-canvas attribution comments
travel with each project. `truth.json` records expected states, placements and
asset hashes. To rebuild from the repository root:

```powershell
.venv\Scripts\python.exe "examples/opencv tools/mosaic_layout_reuse/build_example.py"
```

The tests compare exact integer-reference pixels, coverage and ordering for the
frozen priority cases, source metadata/tickets in both Debug and Lean, and byte
reproduction of every graph, source PNG and sealed layout. Native registration
estimates pixel geometry; these synthetic images do not qualify a physical camera
rig or synchronized acquisitions.
