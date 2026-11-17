# Stabilization border fills with measured coverage

Open a project with **File → Open…**, then Run. Select Fill stabilization borders
for review. Its **Image/Coverage** are current measured data; **Display** adds
neighboring samples only for review. The filmstrip explicitly selects Display
with drawable fill/missing outlines. Preview colors measured pixels green,
filled pixels amber and missing pixels red.

| Project | Expected result | Demonstrates |
| --- | --- | --- |
| [project.json](project.json) | Pass, complete | Known jitter, matching corrected grids and exact neighbor-filled borders. |
| [shifted_grids.json](shifted_grids.json) | Pass, explicitly partial | Pull between differently shifted corrected grids; six corner pixels have no surviving donor. |
| [changed_donor.json](changed_donor.json) | Pass, complete | Reject transfer near a changed patch and use another donor. |
| [highdepth_mean.json](highdepth_mean.json) | Pass, complete | Mean composition retains original BGR uint16 samples above 255. |
| [no_consistent_neighbors.json](no_consistent_neighbors.json) | Fail, no fills | Brighter neighbors fail measured-overlap consistency. |
| [distance_limited.json](distance_limited.json) | Fail, partial review | A one-pixel fill limit leaves requests incomplete under the strict policy. |
| [native_stabilization.json](native_stabilization.json) | Pass, explicitly partial | Fit motion natively, then fill available borders while retaining small uncaptured corners. |

The partial passing projects allow missing requested pixels and require at least
98% fill. The other projects require every requested hole to be filled. A failed
availability judgment retains measured and review images. Debug and Lean have
the same numerical outcomes; Lean omits optional drawings and Preview.

## Independent controls and source rights

Images, generator, trajectories, projects and controls are original work by
OpenCVDataflowGUI contributors under the repository's [MIT license](../../../LICENSE).
There are no customer or third-party images. Every project has an attribution
comment. This is a focused OpenCV tool fixture, using controlled synthetic data.

An independently constructed larger world contains a slow intensity ramp with a
textured center. Three **160×112** crops use offsets **(−4,−2), (0,0), (+4,+2)**
with an eight-pixel margin. Known corrections map the raw crops to a common
scene reference. Shifted grids add horizontal offsets **+1, −2, +2** to those
corrected canvases and retain their inverse shifts as spatial metadata. All
fitted or known transforms keep their direction and measured support explicit.

The reference known-grid target has **760** uncovered pixels. The one-pixel
distance limit fills **266**, leaving **494**. The shifted-grid target requests
**650** pixels; **644** can be filled and **six** lower-left corner samples were
cropped out of every available corrected donor. Those stay missing. The native
fitted case qualifies at least 98% available fill and at most eight missing
pixels on this fixture, without requiring a fabricated complete canvas.

High-depth sources store original values multiplied by **257, 180, 120** in BGR
channels. Known corrected samples and fills match independent world pixels
exactly. Native fitting has its own reference check: filled boundary samples
match the independently evaluated world ramp within **two 8-bit units**, allowing
compound nearest-neighbor rounding. That tolerance is specific to this fixture.
Tests also verify bit-identical current measured pixels, unchanged masks/tickets,
absence of measured/fill overlap, donor provenance and intentional rejection.

[truth.json](truth.json) records geometry, case policies and hashes for all source
images and sealed trajectories. The generator constructs truth and trajectories
without fitting motion or executing the fill tool. The native project uses
OpenCV optical flow and motion fitting at run time instead of a stored fit.
These controls do not qualify changing real scenes or physical camera motion.

Run or regenerate from the repository environment:

```powershell
python -m opencv_dataflow.assistant_cli run "examples/opencv tools/stabilization_fill/project.json" --profile debug --summary
python "examples/opencv tools/stabilization_fill/build_example.py"
```

Use [the contracts](../../../docs/stabilization-fill.md) for alignment assertions,
full interpolation support, alpha, consistency gates, selection, distance and
resource limits. Inspect with Image/Coverage, rather than treating Display as a
new capture of the current part.
