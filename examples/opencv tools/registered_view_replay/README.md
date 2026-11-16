# Registered views on a common canvas

Open a project below with **File → Open…**, then choose **Run all**. Each source
loads its two or three images, Batch waits for the complete set, and Replay
registered views places each image independently on the saved grid. Inspect Image
for the preview, Frames for the group, Coverage for individual support, Common
for shared valid pixels and Info for transforms and tickets.

| Project | Expected final result |
| --- | --- |
| [union.json](union.json) | Pass; three separate color frames on the union canvas, with invalid borders zeroed. |
| [intersection.json](intersection.json) | Pass; exact common support including circular mask holes; Unbatch runs selected statistics on each ticket. |
| [native_union.json](native_union.json) | Pass; native placement matches Union's uint8 images exactly. |
| [native_intersection.json](native_intersection.json) | Pass; native Crop factory and wrapper mask handling match Intersection. |
| [reordered_missing.json](reordered_missing.json) | Pass with incomplete status; Right/Middle retain received order, full canvas and filename tickets. |
| [depth16.json](depth16.json) | Pass; common-crop grayscale values remain uint16. |
| [no_common.json](no_common.json) | Deliberate fail; disjoint saved geometry has no common support, so no frames reach Unbatch/statistics. |
| [budget_guard.json](budget_guard.json) | Deliberate validation error; a 1 MiB work budget rejects replay before warping. |
| [native_depth_guard.json](native_depth_guard.json) | Deliberate validation error; native detail refuses uint16 instead of narrowing. |

Three 192×128 captures occupy known x offsets 0, 48 and 96 on a 288×128 grid.
Brightness changes make separate results visible. Layouts contain independent
known placements; replay performs no feature fitting or blending. No-common uses
a separate disjoint 576×128 layout. Exact hashes pin all three layout assets.

Frames → Unbatch preserves per-frame tickets. Intersection projects also connect
Common to the statistics mask so invalid borders/holes cannot bias means or
percentiles. Preview frame only selects the displayed entry. Frame order can use
received or calibration order. Crop origins and `source_to_frame` map results to
the full canvas and original source images.

Images, layouts, generators and projects are original work by OpenCVDataflowGUI
contributors under the repository's [MIT license](../../../LICENSE). No customer
or third-party images are included. Each graph carries an attribution/scope note.
`truth.json` records expected states, placements and all nine asset hashes.
Rebuild from the repository root:

```powershell
.venv\Scripts\python.exe "examples/opencv tools/registered_view_replay/build_example.py"
```

See the [block contract](../../../docs/registered-view-replay.md) for support,
timestamps, native restrictions and memory guards. Synthetic images do not
establish physical camera geometry or synchronized exposure. For fitting or
reusing a layout first, see [mosaic layout reuse](../mosaic_layout_reuse/README.md).
