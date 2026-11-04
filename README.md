# OpenCVDataflowGUI

Develop and run vision applications by connecting extensible Python blocks in a
desktop node editor. Built with OpenCV, NumPy, and PySide6.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Architecture](#architecture)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

The goal is quick, visual development of OpenCV inspection applications without
writing code for each pipeline. Custom Python blocks extend the same contracts as
built-in tools. Workflow references include Cognex QuickBuild's tool blocks and
MVTec MERLIC's connected tool flow; this is an independent early-stage project.

## Install

Requires Python 3.11 or newer and a desktop OS supported by PySide6 and OpenCV.
Python 3.13 is the initial Windows development environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

On Linux/macOS use `.venv/bin/python` instead of `.venv\Scripts\python`.

## Usage

Start the editor from the repository's virtual environment:

```powershell
.\.venv\Scripts\opencv-dataflow
```

Alternatively, run `python -m opencv_dataflow.app` from an environment with the
package installed. The editor opens a synthetic inspection demo; press **Run once**
(F5) to see two detected contours and a passing result. No camera is needed.

![Editor showing the synthetic inspection demo](docs/images/editor.png)

1. Expand a type and subtype in the left toolbox tree, then double-click a tool to add a block.
2. Drag between an output socket and an input socket in either direction. Image
   ports are cyan, numeric ports amber, and structured-data ports purple.
3. Select a block, edit parameters in the right inspector, and click **Apply
   configuration**. Choose exposed ports and the optional node thumbnail there.
   OpenCV enum controls use dropdowns; booleans use checkboxes. Hover over controls
   with help text for method-specific guidance.
4. Run once or continuously. Inspect retained input, output, and graphic images
   with the preview selector. Nodes show pass/fail/error state, timing, and score
   intensity where provided. A failed inspection still supplies results downstream;
   execution errors skip dependent blocks.
5. Save/load JSON projects from File. Select a wire or block and press Delete to
   remove it. Use the wheel to zoom, middle-drag to pan, and **Fit graph** to frame it.

Node rearrangements mark the project as modified. **Save/Ctrl+S remembers node
positions even when connections are unchanged**, including layouts inside nested
ToolBlocks. Layout-only moves retain existing results. Positions are saved in the
project; there is no automatic save on movement.

Load trusted custom blocks via **File > Load Python plugin**. Start with
[`examples/custom_invert.py`](examples/custom_invert.py). Plugins run arbitrary Python;
project files never automatically import them. Load the same plugins before opening
projects that use their types. No code is needed for built-in pipelines.

Built-in tools include demo/image/camera inputs; G-API grayscale, Gaussian blur,
and Canny; threshold; contour inspection with overlay and mask images; image saving;
and structured result output to CSV, XML, or SQLite. Set a Write results block's
`format` to `csv`, `xml`, or `sqlite` and connect a `data` output to it. CSV and
SQLite append; XML and image writers reject existing files. Output blocks write
on every run, including continuous runs. Relative paths resolve from the process's
working directory; use absolute paths for portable local execution expectations.

Additional tools include monochrome/per-channel color histograms, grid-based planar
pixel-to-physical mapping, offline multi-view lens calibration, undistortion, white/
color balance, reusable calibration ToolBlocks, and typed depth-image loading,
RGB-D pairing and depth sampling. Calibration profiles can be saved and loaded as
JSON for subsequent measurement jobs without a grid in every image. See
[calibration and depth workflows](docs/calibration-depth.md) for units, origin and
coordinate-space contracts. These are offline tools; no live robot calibration or
point-cloud processing is implemented.

### Build and test a starter project

For the real-image Lena review example, run `python scripts/prepare_lena.py` once,
then open `examples/lena.json`. The optional, hash-checked image is not committed
or included under this project's MIT license; see [asset provenance](examples/assets/README.md).
This example includes grayscale/blur, Canny, color/grayscale histograms and template
matching, including an editable nested ToolBlock. Its image input uses a
project-relative path, so move its `assets` folder together with its JSON file.

Open **File > Load starter inspection**, or launch it directly:

```powershell
.\.venv\Scripts\opencv-dataflow --starter
```

The project uses nine blocks: synthetic parts → crop ROI → grayscale → median
filter → Otsu threshold → morphology → connected-component measurements, with
histogram and Canny branches. Run once to detect two parts at a 100% inspection
score. Change **Measure two parts > Expected count** to 3 and apply it to exercise
an inspection failure, then restore 2. Save the graph as your own project.

The same project is saved in [`examples/starter_inspection.json`](examples/starter_inspection.json).
Open it through File > Open or pass its path to `opencv-dataflow`. A second project,
[`examples/template_matching.json`](examples/template_matching.json), locates a
reference patch at pixel `(250, 85)`; it is also available from File.

To verify a saved graph without the editor:

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/starter_inspection.json --repeat 3
```

Add `--report <new-file.json>` for a JSON receipt of each run, including states,
scores, timings and numeric/data outputs. Existing reports are preserved. Use
`--plugin <trusted-file.py>` to explicitly load custom Python blocks. Exit status
is 0 when all runs pass, 1 for an inspection/execution failure, and 2 for load or
configuration errors. Output blocks still perform their configured writes on every
run. `python -m opencv_dataflow.runner` is an alternative to the console command.

The expanded [base-tool catalog](docs/base-tools.md) covers conversion, crop/resize/
flip/rotate, filters, automatic/adaptive thresholds, color-range masks, morphology,
bitwise operations, blending, gradients, components and template matching. These
new wrappers use eager OpenCV calls alongside the existing native G-API tools.

Additional wrappers include histogram equalization, CLAHE, normalization, gamma,
channel split/merge, masking, absolute difference, corner detection, Hough lines/
circles and distance transform. See [additional tools](docs/additional-tools.md).
After preparing the optional Lena asset, open `examples/lena_features.json` to try
CLAHE → corners, edges → line segments, and a histogram in one project.

The [registration and transform tools](docs/registration-tools.md) add arbitrary
rotation, affine/perspective warps, borders, pyramids, custom filtering, unsharp
masking, FAST/ORB features, descriptor matching, homography estimation and point
transforms. Open `examples/lena_registration.json` after preparing the Lena asset
to match the image to a known rotated copy and compare the estimated warp.

Further tools include box and separable filtering, Harris/minimum-eigenvalue
responses, subpixel corner refinement, phase correlation, full/partial affine
estimation, coordinate remapping, and Cartesian-to-polar image fields. See
[analysis tools and parameter conventions](docs/analysis-tools.md).

The [complete parameter reference](docs/block-parameters.md) lists defaults,
dropdown choices and input ports directly from the block definitions. Existing
wrappers also expose kernel dimensions, anchors, borders, output depths, masks,
threshold modes, contour modes, feature detector controls, matching norms, robust
solver settings, calibration flags and image encoder options where applicable.
Native buffers are managed by the runtime; hardware requests depend on the camera
backend. Existing saved projects acquire new defaults when loaded.

### Masks and regions

Add **Image processing > Regions > Shape region**, choose rectangle, circle or
polygon from its dropdown, and connect its `region` or `mask` output to supported
blocks. Regions retain input-image coordinates and include a tinted overlay.
When both a mask and region are connected, their intersection is used.

Run once, select the Shape region block, then click **Draw / edit region…** in
the inspector. Draw rectangles/circles by dragging or polygons by clicking
vertices. Drag handles to resize, or drag a shape to move it. **Apply region**
updates the block and marks the project modified; **Cancel** preserves it.
The editor uses a captured frame, with correctly scaled pixel coordinates.

Open **File > Load regions example**, or `examples/regions.json`, for a working
10-block project that inspects each synthetic part separately, computes a polygon
histogram and blurs only the circular selection. No image download is required.
Run it headlessly with `opencv-dataflow-run examples/regions.json --repeat 3`.

Selections support filters, segmentation, gradients, image math, histogram and
contrast tools, corner/ORB detectors, contours/components and template search.
Contour/component **Region mode** chooses whole-object center filtering or clipped
measurements. See [region behavior and supported tools](docs/regions.md) for
outside-pixel policies, neighborhood behavior and coordinate rules.

### Image file formats

The **Image file** input loads a file and reformats it using inspector dropdowns:

- **Color format:** unchanged, BGR, RGB, BGRA, RGBA, grayscale, HSV, or Lab.
- **Pixel depth:** unchanged, uint8, uint16, or float32 (depth is per channel).

Choose **unchanged** for both to preserve decoded channels, alpha, and depth.
Defaults remain BGR/uint8 for existing jobs. Integer depth conversion scales the
full range (8-bit 255 maps to 16-bit 65535); integer-to-float maps to [0,1]. Float
sources keep their numeric values in float outputs, and float-to-integer clips
to [0,1] before scaling and rounding. There is no per-image contrast normalization.
HSV and Lab support uint8 or float32, not uint16, and use OpenCV's native channel
ranges. See [image format details](docs/image-formats.md).

The `image` output carries the requested format. The `info` output reports size,
channel count, depth, and source/output formats; the decoded source is also available
in the graphic preview selector. Previews and built-in tools interpret the color
encoding correctly. Processing tools still have their own depth requirements:
native G-API Grayscale and Canny require uint8. For 16-bit or float grayscale, select
grayscale directly in the file input.

### Reusable ToolBlocks

Select nodes and use **ToolBlocks > Create from selected nodes** to encapsulate
them as one reusable block. Crossing wires are preserved, unbound inputs become
public inputs, and terminal outputs become public outputs. Double-click the new
node (or use **Edit internal graph** in its inspector) to develop its contents in
the same node editor. ToolBlocks can contain other ToolBlocks. **Up one level** and
the job breadcrumb show where you are.

Use **Public terminals** to name and map the interface to internal typed ports.
Renaming a terminal preserves connected wires; disconnect a terminal before removing
its binding. Exposed-port checkboxes then choose which public ports appear on the node.
The outer block aggregates child pass/fail status, minimum available score, graphics,
and errors. Internal run records remain inspectable by opening the block after a run.

**Export ToolBlock** writes a `.toolblock.json` definition; **Import ToolBlock**
adds an instance and a session library entry. Creation and export also add reusable
library snapshots. Double-click a library entry to add another independent instance.
Editing an instance does not change other instances or a previously captured library
snapshot; export again to capture the edited version. Job files embed their ToolBlocks,
so they do not depend on the original definition file. Required Python plugins must
still be loaded explicitly.

For a ready-made nested pattern, import
[`examples/inspect_shapes.toolblock.json`](examples/inspect_shapes.toolblock.json)
and connect a Demo image or Image file output to its `image` input.

Run controls always execute the entire root job, including when viewing a nested
ToolBlock. Saving also saves the root job. See [docs/toolblocks.md](docs/toolblocks.md)
for the class model and a worked example.

The Python runtime can also execute graphs independently of the editor:

```python
from opencv_dataflow.blocks import builtin_registry, demo_graph

graph = demo_graph(builtin_registry())
graph.run()
for block in graph.blocks.values():
    print(block.name, block.state, block.error)
graph.close()
```

## Architecture

- `core.py`: block/result contracts, typed ports, acyclic graph scheduling, JSON projects.
- `blocks.py`: acquisition, native G-API processing, inspection, and output sinks.
- `processing.py`: base OpenCV geometry, filtering, segmentation, math and inspection wrappers.
- `features.py`, `registration.py`, `analysis_tools.py`: contrast, features, registration and numerical image tools.
- `regions.py`, `selection.py`: reusable shape regions, mask intersections and selection policies.
- `region_editor.py`: interactive image-space drawing and editing of region shapes.
- `examples.py` and `runner.py`: built-in starter graphs and headless project verification.
- `plugins.py`: explicitly loaded trusted Python extensions.
- `canvas.py`: movable nodes, typed sockets, selectable wires, pan and zoom.
- `app.py`: Qt editor, inspector, worker execution, project and plugin actions.
- `toolblocks.py`: composite blocks, public terminal mappings, reusable definitions.
- `toolblock_editor.py`: public terminal configuration dialog.
- `examples/`: reusable custom content.
- `tests/`: local runtime and persistence checks.

Grayscale, Gaussian blur, and Canny each build a native `cv2.GComputation`.
The Python scheduler connects them with ordinary OpenCV, custom code, and I/O.
The complete editor graph is **not** compiled into one fused G-API computation.
Block histories retain the latest run only; projects store configuration, not
camera frames or execution histories.

See [docs/architecture.md](docs/architecture.md) for execution and extension contracts.
This is a working prototype: live camera hardware is not part of automated
validation, cancellation waits for the current native/custom operation to return,
and there is no operator interface, undo/redo, or full OpenCV tool catalog yet.

## Maintainers

[@votecoffee](https://github.com/votecoffee)

## Contributing

Use a `src/` package layout and keep reusable examples sanitized. Keep local
projects, images, device configuration, and scratch artifacts in ignored `.local/`.
Run local checks before committing:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check src tests examples
.\.venv\Scripts\python scripts/block_parameter_reference.py --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and
[docs/roadmap.md](docs/roadmap.md) for planned work.

## License

[MIT](LICENSE) © 2026 votecoffee.
