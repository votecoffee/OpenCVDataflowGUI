# Simple3DMeasure

Train plane datums and inspect step heights, plane angles, signed probe
distances and surface residual ranges. Requires the optional
`opencv-simple-3d-measure` plugin **0.1.0.dev0** and the host integration.

Install the wheel/source package in the Python environment running the app,
then restart the app and its runners:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps C:\path\to\opencv_simple_3d_measure-0.1.0.dev5-py3-none-any.whl
```

| Project | Evidence |
| --- | --- |
| [Known step](generated/good/project.json) | Independent noisy sampling of a generated 5 mm step. |
| [Raised feature](generated/raised_feature/project.json) | High points are excluded from plane fitting but still make surface range fail. |
| [Missing surface](generated/missing_surface/project.json) | Missing height/angle/range stay unavailable. A valid datum probe remains available. |
| [Real T-LESS captures](real_tless/README.md) | Three measured first-return captures with explicit supplied annotation fixtures and accuracy limits. |
| [Line measurements](lines/README.md) | Five generated controls and three measured T-LESS section examples, including missing/short support and a retained angle-guard failure. Requires dev2. |
| [Minimum-zone flatness](flatness/README.md) | Four generated controls and three real T-LESS captures distinguish all-selected minimum-zone width from datum residual range. Requires dev3 and optional SciPy. |
| [Retained source context](reference_context/README.md) | Real T-LESS RGB/pixel/calibration context and source appearance review with unchanged earlier plane values. Requires dev4. |
| [Residual review](residuals/README.md) | Seven unchanged generated/real cases with fixed-band residual colors, histograms and full source-row exports. Requires dev5. |

Run a project and select **Simple3DMeasure → Cloud / fit**. Its thumbnail and
popout use the native flyaround point-cloud viewer. Green shows fitting
inliers, red exclusions, magenta unavailable region samples, gray scene,
blue plane grids, violet trained region boxes, yellow dimensions/angle arcs
and RGB datum axes. Blue datum Z follows the signed normal. Grids, boxes and
dimension lines are derived graphics. Inspect **measurements**, **planes** and
**report** for values, units, limits, fit coverage and extrapolation evidence.

With plugin **0.1.0.dev1** and the native teaching host API, open
**Train 3D measurements…** to restore the retained training cloud or teach a
new reference. Draw regions, drag XYZ face handles, orbit/zoom/pan or focus a
selected region, pick probe points and define measurements/limits. Green/red
show cached fit inliers/exclusions; cyan shows untrained selected points.
Edits share Undo/Redo and invalidate staging until the current draft trains.
See the [training guide](TRAINING.md).

**Review trained / completed run…** opens cached native evidence.
Choose a region or toggle graphic layers. After a Debug execution,
**Capture completed run** freezes scene and fixture from that completed
execution. **Export visible review…** saves the selected native PLY, raw report,
hashes, a portable recipe and capture identity. **Import recipe…** loads a CLI/API-trained
`.s3m`; **Stage recipe** edits the inspector draft and **Apply** activates it.
Cancel preserves the active parameters. Older hosts explicitly fall back to
recipe review/import; update the host to use graphical teaching.

With **dev2**, choose **New line region** before drawing. Add guarded parallel
separation, acute line angle, point-to-line distance or observed axial span.
The new `lines` output and Lines graphic layer retain axis/span/radial evidence.
**Flip axis** changes the retained direction. [Definitions and compatibility](LINES.md).
Plane-only archives and numerical reports remain compatible; line archives
require dev2. A fitted line is not automatically a detected physical edge.

With **dev3**, add **Minimum-zone flatness (all selected)** on a plane region.
Zones/Supports layers show the enclosing planes and source witnesses, including
points excluded from the datum fit. Geometry budgets or failed checks make it
unavailable; there is no least-squares substitute. [Details and dependencies](FLATNESS.md).

With **dev4**, source RGB, original pixel rows/size and bounded JSON context
can travel with the trained reference. Toggle source colors in training or
switch cached review to source appearance. Completed captures retain fixture/
execution identity; exports include portable source NPZ when context is present.
Metadata remains descriptive and does not recalibrate points.
[Context conventions and limits](REFERENCE_CONTEXT.md).

With **dev5**, choose **Residuals / all selected** in native review, then select
a region for signed plane/radial line distances, physical histograms and tail
counts. Runtime `residual_graphics` opts into the native `residuals` cloud and
`residual_report`; graphics-off skips the additional work. Fit bands and
measurement reports stay unchanged. [Definitions and exports](RESIDUALS.md).

`surface_range` is peak-to-valley residual over **all selected samples** about
the robustly fitted least-squares plane. `minimum_zone_flatness` is the separate
smallest parallel-plane enclosure, with optional SciPy and bounded hull work.
Step height uses a signed target inlier centroid with a 5-degree parallel
guard. Missing/degenerate fits produce unavailable measurements, never zeros.
Turn off `render_graphics` to skip graphic composition without changing values.

Rebuild into fresh folders with the plugin and host installed:

```powershell
python build_demo.py new_generated --host-projects
python build_tless.py real_tless/source new_tless
```

Generated surfaces use the MIT-licensed builder and contain one visible surface
per XY sample; they are numerical controls. Real raw/derived T-LESS material
uses CC BY 4.0 with [author attribution](real_tless/ATTRIBUTION.md).
The manifest pins sample assets; PLY display sampling is separate from raw
numerical measurement. These examples do not establish production accuracy.
