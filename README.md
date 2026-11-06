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

OpenCV 5's **contrib** distribution is required. When upgrading an existing
environment, remove the old OpenCV distributions before reinstalling; they share
the `cv2` namespace and must not coexist:

```powershell
.\.venv\Scripts\python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

## Usage

**3D vision** includes camera and pose geometry, stereo calibration/rectification,
disparity, RGB-D projection and registration, point clouds, ICP/PPF, ArUco,
structured light, odometry, and posed TSDF fusion. Try [stereo cloud](examples/3d/stereo_cloud/stereo_cloud.json), press **Run once**, and open **Results > 3D point-cloud viewer…**.
The example reconstructs a synthetic 3 m plane; no hardware or extra packages
are needed. The [stereo confidence example](examples/3d/stereo_confidence/README.md)
uses a WLS confidence cutoff before reconstruction. The
[depth hole filling example](examples/3d/depth_hole_filling/README.md) restores
small enclosed gaps and marks the estimated samples.
See [3D workflows and conventions](docs/vision-3d.md).

**DNN/ML tools** are available under **ML** in the toolbox (model writers under **Output**): ONNX inference,
classification, embedding conversion, SIFT + LightGlue matching, and Annoy vector
index building/search/save/load. Try [DNN embedding](examples/ml/dnn_embeddings/dnn_embeddings.json) with the
included tiny synthetic ONNX model. [Annoy search](examples/ml/annoy_search/annoy_search.json) and [LightGlue matcher](examples/ml/lightglue_match/lightglue_match.json) use optional libraries. See [ML setup and workflows](docs/ml.md)
for installation, local weights and runnable examples. Graph execution never downloads
learned weights or imports optional ML libraries until their blocks need them.

Use [OCR and text verification](docs/ocr.md) to detect text regions, read
cropped lines with a local CTC model, and check OCR or barcode results.

The optional `barcodes` extra adds [Data Matrix and multi-format barcode reading](docs/barcode-reading.md).

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
   ports and wires are gold for images, purple for regions/masks, blue for poses/transforms,
   pale gold for pyramids, turquoise for depth, green for numbers, and slate for other data.
   The editor checks the declared kinds and accepts supported conversions. For example,
   a numeric count can feed a structured writer, and a pyramid can feed an image input.
   Hover over a converted wire to see its rule. See [connection compatibility](docs/architecture.md#connection-compatibility).
   Hover over a connector dot or its label to see the full port name and any captured
   number, boolean, or text value.
   The **Debug** run profile captures input values.
3. Select a block, edit parameters in the right inspector, and click **Apply
   configuration**. Switching blocks discards unapplied edits. Use **Show advanced**
   to configure recipe overrides, ports, and node display. Block size and the
   result graphics setting take effect immediately.
   Hover over a field for its limits. The inspector shows an inline message when an entry is invalid.
   **Run block** previews eligible tools with their last captured inputs, including text and
   other structured data. The [parameter reference](docs/block-parameters.md) lists declared limits.
   On the canvas, **!** marks invalid configuration or input and **×** marks an
   execution exception. Hover over the node for details.
   Add a plain-text **Description** to explain the block in its hover tooltip.
   Use **Edit documentation…** for longer, formatted notes stored with the block.
   OpenCV enum controls use dropdowns; booleans use checkboxes. Hover over controls
   with help text for method-specific guidance.
4. Run once or continuously. Inspect retained input, output, and graphic images
   with the preview selector. Nodes show pass/fail/error state, timing, and score
   intensity where provided. A failed inspection still supplies results downstream;
   execution errors skip dependent blocks.
   The inspector and automatic node thumbnails prefer the result image, then an output image.
   Choose an output or graphic and click **Pop out…** to watch it in a separate
   window with a status-colored border and the ticket in its title.
   In the **Debug** run profile, pattern matching, template instances, oriented
   edges, and collection filtering include review graphics with accepted items
   in green and rejected candidates in red.

To combine annotations on a final image, expose **Result graphics** on each
source block and connect those outputs to **Overlay result graphics**. Connect
the final image to its **image** input, then run in **Debug**. Use **Choose
graphics…** on the overlay block to check individual layers, then **Apply
configuration** and run again. Image-aligned layers follow crop, resize, and
other recorded transforms in the same coordinate frame. Result graphics are
drawing objects. The overlay block draws them onto the final image to produce
the result image. Standalone previews such as histograms remain available for
inspection. [Try the overlay example](examples/inspection/overlay_result_graphics/README.md).

The overlay block starts with two visible graphics inputs. Expose more in the
inspector to connect up to eight sources. A ToolBlock's **Result graphics**
collects drawings from its internal blocks. You can also map an internal block's
exposed graphics to a separate public terminal.
5. Use **File → Open…** for saved projects and the [bundled examples grouped by workflow](examples/README.md).
   Select a wire or block and press Delete to remove it. Use the wheel to zoom,
   middle-drag to pan, and **Fit graph** to frame the graph.

Use **File → Export program documentation…** to generate a self-contained HTML
guide from the current applied configuration. It includes a simplified graph,
a linked block index, wiring and port tables, applied parameters, and each
block's description and rich-text notes. Nested ToolBlocks and parallel
inspection templates include their own diagrams and public terminal mappings.
Open the HTML in a browser to read it offline or print it to PDF. The export
includes configured values such as file paths and retains basic note formatting.

Use **File → Export project XML…** to snapshot the open project's saved fields and
values, including in-memory edits that have been applied. The export includes
nested ToolBlocks, recipes, saved training data, descriptions, and layout. External
assets are recorded by their saved references. Use **File → Compare project XML…**
to inspect added, removed, and changed values alongside changes to block order.
The comparison matches blocks by their saved IDs and shows before and after
values. JSON remains the editable project format.

Select a block and use **Save block XML…** to export its applied configuration
with a type-specific suffix such as `.cv.morphology.xml`. **Load block XML…**
applies the file to a selected block of the same type, retaining the destination
node's ID, position, and compatible connections. If loading would remove wires,
the inspector lists them and asks before proceeding. The XML includes saved
training data, documentation, port settings, and any nested block definitions.

After **Run once** captures inputs, click **Run block** in the inspector to preview
a supported block once with its current parameter edits. Enable **Live results**
to preview automatically. Edits within 300 ms of one another produce a single
preview, and **Run block** can run that preview immediately. The checkbox starts
off and resets when you leave the block. Previews stay in the inspector. Use
**Apply configuration** to save the edits to the graph.

In a **Morphology** block, choose **Edit kernel…** to preview a rectangle,
ellipse, cross, or diamond, or toggle individual cells in a custom kernel.
**Use kernel** stages the edit for live preview and **Apply configuration**
keeps it in the graph.

Node rearrangements mark the project as modified. **Save/Ctrl+S remembers node
positions even when connections are unchanged**, including layouts inside nested
ToolBlocks. Layout-only moves retain existing results. Positions are saved in the
project; there is no automatic save on movement.
Choose **Small**, **Medium**, or **Large** under **Block size** in the inspector.
The canvas applies the size immediately and keeps room for every exposed port.
Small hides the thumbnail, and Large gives it more room. Save the project to keep
the size, including for nodes inside ToolBlocks.
Use **Workspace → Auto arrange blocks** to organize the current canvas from
inputs on the left to outputs on the right. Groups and independent flows get
separate lanes. Save the project to keep the new positions.

Node borders indicate tool category (including purple region tools and blue pose
tools); node headers keep their pass/fail/error status colors and score intensity.
Selection adds a white outline. **Workspace > Color legend…** explains the palette;
see [canvas colors](docs/canvas-colors.md) for typing and custom-block details.

For larger workspaces, select nodes and use **Workspace > Group selected nodes**
(Ctrl+G). Drag the group's dashed header/border to move its members together;
**Ungroup selection** (Ctrl+Shift+G) removes the container. The overview above the
inspector in the upper-right corner shows the visible region. Click or drag it to
navigate. Save remembers groups, zoom, and viewport position, including inside ToolBlocks.

Open the [image-set workspace](examples/inputs/image_set_workspace/image_set_workspace.json)
for a grouped, three-image inspection.
Press **Continuous** to run the complete set and see filename tickets and verdict
borders in **Results > Image results filmstrip**. In an **Image file** block, use
**Select images…** or **Select directory…**, then **Apply configuration**. Run once
advances one image; Continuous stops at the end unless `repeat=loop`.
**File > Reset flow state** restarts the sequence. Filmstrips also appear in the
validation-set window. [Workspace and image-set details](docs/workspaces-image-sets.md)
cover ordering, queue behavior, filename identity, retention and Lean mode.

Train geometric and texture patterns with **Match shape pattern** and **Match
texture pattern**. Select a matcher, run once to capture its input, then click
**Train from input** in the inspector. Models are retained in saved projects and
can also be imported/exported as JSON. Separate **Train shape pattern** and
**Train texture pattern** blocks provide reusable wired models. Geometric features
include edges, corners, lines and circles; texture detection supports ORB and SIFT.
**Find oriented edges** selects vertical, horizontal or angled edges and filters
thickness, continuous length, strength, intensity, contrast and coverage.

Open [trained patterns and edges](examples/pattern_matching/trained_patterns/trained_patterns.json) and press F5.
It needs no external assets: expected results are one geometric match, one texture
match and two vertical edges. See [training and edge workflows](docs/patterns-edges.md)
for training regions, pose coordinates, controls and current algorithm limits.
Use **Subpixel** on template match blocks to refine centers and poses.
[Compare pixel-grid and subpixel matches](examples/pattern_matching/subpixel_template/README.md).

The [SIFT registration example](examples/pattern_matching/sift_registration/README.md) detects and
matches keypoints before estimating an image homography. The
[AKAZE registration example](examples/pattern_matching/akaze_registration/README.md) runs the
same workflow with binary MLDB descriptors. The [FLANN registration
example](examples/pattern_matching/flann_registration/README.md) uses approximate KD-tree matching.
The [KAZE registration example](examples/pattern_matching/kaze_registration/README.md) pairs
extended KAZE descriptors with the same matcher.
The [BRISK registration example](examples/pattern_matching/brisk_registration/README.md) uses
binary descriptors and Hamming distance. The [descriptor radius example](examples/pattern_matching/descriptor_radius/README.md)
retains multiple candidates per query keypoint within a distance limit.
The [MSER example](examples/image_processing/mser_regions/README.md)
shows stable bright and dark regions in an image.

Pattern matchers also return a configurable **pose**: set local x/y/angular offsets
to locate an inspection origin on the part. **Pose from feature** creates a frame
from a ranked blob, line, contour or pattern match, and **Offset pose** composes
another local offset. Connect a pose to **Region relative to pose** to position a
rectangle, circle or polygon for downstream masked inspections. Try [pose-relative inspection](examples/inspection/pose_inspection/pose_inspection.json); see
[pose-relative inspection](docs/poses.md) for coordinates and a complete workflow.

**Shape region**, **Region relative to pose** and **Shape mask** have visual
rectangle/circle/polygon editors in their inspectors. Relative-region and mask
editors map drawing and handle movements into the supplied pose's local frame.
Shape mask can also operate in ordinary image pixels with its pose disconnected.

Input sources carry numeric, string or timestamp tickets, generated automatically
or supplied through an external data input. **Control flow** tools split/join
parallel branches, batch/unbatch entries, select cases and restore FIFO by ticket.
Try [ticketed control flow](examples/flow_control/control_flow/control_flow.json). See
[tickets and control flow](docs/control-flow.md) for external/PLC ticket inputs,
buffer limits and stream execution semantics.

Ticketed connections carry each tool's pass/fail, score and accumulated failure
status. Every input has a **continue / reject_failed** policy; rejection emits no
data and retains the last processed previews. **Final ticket judgment** assigns an
explicit decision separately from intermediate tool results. Open **Results >
Ticket results…** to search tickets and inspect their status and event timelines.
Try [ticket results](examples/flow_control/ticket_results/ticket_results.json). See [ticket results](docs/ticket-results.md)
for status propagation, final decisions, retention limits and plugin event data.

**Coordinate inputs** offers ticket-matched waiting with timeouts, latest snapshots,
triggered latest snapshots, and first-arrival-per-ticket selection. Freshness,
duplicate and late-arrival controls make reuse explicit. Try [input coordination](examples/flow_control/input_coordination/input_coordination.json). See [input coordination](docs/input-coordination.md) for
recommended policies and cooperative timing limits.

**Recipes > Manage recipes…** stores versioned settings and trained pattern models.
Use **Select recipe** / **Recipe value or model** blocks, or bind tool parameters
and matcher models in **Recipe overrides**. Images retain their recipe revision
across queues and global changes. Try [recipes](examples/flow_control/recipes/recipes.json); see
[recipes and model swaps](docs/recipes.md) for activation and persistence rules.

**Inspect each pose in parallel** runs a reusable inspection ToolBlock for every
pattern match, measured feature or supplied pose, then returns one aggregate. Set
worker count and optional N-pass/N-failure stop thresholds; results explicitly
identify uninspected parts. The [parallel inspection example](examples/flow_control/parallel_parts/parallel_parts.json) finds
and inspects 25 parts. See [parallel part inspection](docs/parallel-inspection.md)
for editing the inspection, acceptance policies and cancellation behavior.

Insert **Queue input images** after a source to retain frames and release them
one at a time. It supports bounded FIFO storage, explicit overflow policies and
optional background acquisition during inspection. Try [queued input](examples/flow_control/queued_input/queued_input.json), then Continuous. See [input queues](docs/input-queues.md) for capacity,
metadata pairing and Stop/Resume behavior. Stop keeps the last completed result visible.

The supported dependency is OpenCV 5 contrib (development checks use 5.0.0.93).
This desktop application uses native Python bindings. Ordinary OpenCV blocks are
the defaults; legacy G-API blocks remain under **Advanced > G-API**.
Migration details and a possible browser/WebAssembly backend are covered in the
[backend notes](docs/opencv-backend.md).

Load trusted custom blocks via **File > Load Python plugin**. The
[custom invert project](examples/user_tools/custom_invert/README.md) demonstrates custom
fields, validation, and a popup editor. Plugins run arbitrary Python. Load the same plugins before opening
projects that use their types. No code is needed for built-in pipelines.

The [geometric inspection applications](applications/README.md) bundle a tray grid,
a needle row, and weighted part discrimination with pinned plugin recipes and
expected counts for headless replay.

Plugin blocks can add their own inspector button and open a Qt dialog for
configuration or review. The dialog can use copies of cached images with their
coordinate metadata. Plugins can opt into a modeless popup to capture new
outputs while the graph runs and subscribe to completed output notices.
Dialogs can also capture the selected block's wired or last processed input.
The reusable image picker gives plugins a source and pyramid-level selector.
Accepted edits appear in the inspector draft for preview and **Apply
configuration**. [Plugin dialog API](docs/plugin-dialogs.md)

If a block raises a Python exception during processing, the app records an error
and traceback in **Diagnostics**. The block emits no outputs, dependent blocks
that require its output are skipped, and independent branches can finish. Plugins
run in the app's process, so native crashes and forced process termination still
end the app.

Custom blocks can supply named `GraphicDrawing` objects through
`Result.drawings`. Each drawing identifies its source image input and contains
objects such as outlines, markers, and labels. The block's existing raster
previews remain available separately.

Built-in tools include demo/image/camera inputs; ordinary grayscale, Gaussian blur,
and Canny; threshold; contour inspection with overlay and mask images; image saving;
and structured result output to CSV, XML, or SQLite. Set a Write results block's
`format` to `csv`, `xml`, or `sqlite` and connect a `data` output to it. CSV and
SQLite append; XML and image writers reject existing files. Output blocks write
on every run, including continuous runs. Relative paths resolve from the process's
working directory; use absolute paths for portable local execution expectations.

The [K-means example](examples/ml/kmeans/README.md) shows color-coded pixel clusters,
an image quantized to the cluster centers, and a scatter plot of clustered numeric samples.
The [watershed example](examples/image_processing/watershed/README.md) separates touching parts
using automatic distance peaks or editable point seeds. A separate block accepts
foreground and background seed masks.

The [histogram thresholds example](examples/inspection/histogram_thresholds/README.md) wires
measured cutoffs into lower-tail, middle-range, and upper-tail masks. Histograms
also report per-channel mean, median, standard deviation, and extrema.

The [Lab color classification example](examples/inspection/lab_color_classification/README.md)
compares a measured patch with named references and reports the closest color.

The [flat-field correction example](examples/image_processing/flat_field/README.md) uses bright and dark
reference images to remove uneven illumination before inspection. The
[guided filter example](examples/image_processing/guided_filter/README.md) reduces noise while
preserving a high-contrast edge. The [domain transform example](examples/image_processing/domain_transform_filter/README.md)
compares three edge-preserving smoothing modes. The
[Bayer demosaicing example](examples/inputs/bayer_demosaicing/README.md) reconstructs
RGB from a synthetic 16-bit sensor mosaic. The
[in-memory codecs example](examples/outputs/in_memory_image_codecs/README.md) sends a
lossless 16-bit RGBA PNG packet between blocks. The
[TIFF page sequence example](examples/inputs/tiff_page_sequence/README.md) replays
a multipage TIFF with separate tickets for each page. The
[multipage TIFF output example](examples/outputs/multipage_tiff_output/README.md) batches
the pages and writes them to a new TIFF file. The
[raw YUV example](examples/inputs/raw_yuv/README.md) decodes an NV12 frame from a file
and from a packed buffer in the graph. The
[EXIF orientation example](examples/inputs/exif_orientation/README.md) compares
metadata-only loading with an explicitly oriented JPEG.

The [skeleton pruning example](examples/image_processing/skeleton_pruning/README.md) extracts
centerlines from a binary mask and removes short spurs. Install the optional
dependency with `python -m pip install -e ".[morphology]"`.

Additional tools include monochrome/per-channel color histograms, planar
pixel-to-physical mapping, offline multi-view lens calibration, undistortion,
white/color balance, reusable calibration ToolBlocks, and typed depth-image
loading, RGB-D pairing and depth sampling. Calibration grid and Calibrate lens
accept chessboards and symmetric or staggered circle targets. Save and load
calibration profiles as JSON for subsequent measurement jobs.
[Calibration and depth workflows](docs/calibration-depth.md) ·
[Circle grid example](examples/calibration/circle_grid_calibration/README.md)

**Check chessboard quality** measures edge width, contrast, and brightness
before forwarding accepted calibration images.
[Checkerboard quality example](examples/calibration/chessboard_quality/README.md)

**Build lens correction maps** creates reusable pinhole correction maps
for **Remap image coordinates**, with output intrinsics and a separate
coordinate frame for the corrected image.
[Lens correction maps example](examples/calibration/lens_correction_maps/README.md)

**Pack remap coordinates** prepares a compact map for **Remap with packed
coordinates**. Lens maps retain their rectified frame. The
[packed remap example](examples/calibration/packed_remap/README.md) compares both paths.

**Inspect lens map displacement** shows correction strength and direction
and reports how much of the output samples inside the source image.
[Lens map displacement example](examples/calibration/lens_map_displacement/README.md)

**Save lens correction maps** and **Load lens correction maps** preserve
the map pair, output intrinsics, and pixel-grid metadata in one file.
[Lens map files example](examples/calibration/lens_map_files/README.md)

**Calibrate lens** reports per-view reprojection errors, board poses,
and estimated parameter uncertainty. A worst-view limit can catch a poor
view even when the overall fit passes.
[Lens fit report example](examples/calibration/lens_fit_report/README.md)

**Validate lens calibration** checks a fixed profile on a separate grid
image and shows the measured reprojection residuals.
[Lens validation example](examples/calibration/validate_lens_calibration/README.md)

**Detect ChArUco board** pairs identified image corners with physical
board coordinates. A partial view can feed the existing PnP pose block.
[ChArUco board example](examples/calibration/charuco_board/README.md)

**Calibrate lens** can also fit a batch of ChArUco observations with
different visible corners in each view. It reports the point count and
reprojection RMS for each image.
[ChArUco lens calibration example](examples/calibration/charuco_lens_calibration/README.md)

**Validate lens calibration** also accepts identified ChArUco points from
a separate partial view and reports residuals by board corner ID.
[ChArUco validation example](examples/calibration/validate_charuco_calibration/README.md)

**Calibrate hand–eye** fits a camera-to-gripper transform from paired
robot and target poses. The output composes with the current gripper
pose to place the camera in the robot base frame.
[Hand–eye example](examples/calibration/hand_eye_calibration/README.md)

### Build and test a starter project

For the real-image Lena review example, run `python scripts/prepare_lena.py` once,
then open `examples/getting_started/lena/lena.json`. The optional, hash-checked image is not committed
or included under this project's MIT license; see [asset provenance](examples/assets/README.md).
This example includes grayscale/blur, Canny, color/grayscale histograms and template
matching, including an editable nested ToolBlock. Its image input uses a
project-relative path, so move its `assets` folder together with its JSON file.

Open [starter inspection](examples/getting_started/starter_inspection/starter_inspection.json), or launch it directly:

```powershell
.\.venv\Scripts\opencv-dataflow --starter
```

The project uses nine blocks: synthetic parts → crop ROI → grayscale → median
filter → Otsu threshold → morphology → connected-component measurements, with
histogram and Canny branches. Run once to detect two parts at a 100% inspection
score. Change **Measure two parts > Expected count** to 3 and apply it to exercise
an inspection failure, then restore 2. Save the graph as your own project.

The same project is saved in [`examples/getting_started/starter_inspection/starter_inspection.json`](examples/getting_started/starter_inspection/starter_inspection.json).
Open it through File > Open or pass its path to `opencv-dataflow`. A second project,
[`examples/pattern_matching/template_matching/template_matching.json`](examples/pattern_matching/template_matching/template_matching.json), locates a
reference patch at pixel `(250, 85)`; it is also available from File.

To verify a saved graph without the editor:

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/getting_started/starter_inspection/starter_inspection.json --repeat 3
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
new wrappers use eager OpenCV calls; legacy native G-API tools live under Advanced.

Additional wrappers include histogram equalization, CLAHE, normalization, gamma,
channel split/merge, masking, absolute difference, corner detection, Hough lines/
circles and distance transform. See [additional tools](docs/additional-tools.md).
After preparing the optional Lena asset, open `examples/image_processing/lena_features/lena_features.json` to try
CLAHE → corners, edges → line segments, and a histogram in one project.

**Find Hough template** locates translated copies of a shape from its edges.
The [generalized Hough example](examples/pattern_matching/generalized_hough/README.md) compares
full-image and region-limited searches.
The [Guil example](examples/pattern_matching/generalized_hough_guil/README.md) finds the same
template after rotation and scaling.

The [color normalization example](examples/image_processing/normalize_color/README.md) compares
shared and independent channel scaling in a selected patch while preserving the
pixels outside it.
The [linear channel transform example](examples/image_processing/channel_transform/README.md)
mixes color channels using a matrix and derives a grayscale difference image.

The [registration and transform tools](docs/registration-tools.md) add arbitrary
rotation, affine/perspective warps, borders, pyramids, custom filtering, unsharp
masking, FAST/ORB features, descriptor matching, homography estimation and point
transforms. Open `examples/pattern_matching/lena_registration/lena_registration.json` after preparing the Lena asset
to match the image to a known rotated copy and compare the estimated warp.

Further tools include box and separable filtering, Harris/minimum-eigenvalue
responses, subpixel corner refinement, phase correlation, full/partial affine
estimation, coordinate remapping, and Cartesian-to-polar image fields. See
[analysis tools and parameter conventions](docs/analysis-tools.md).
**Iterative phase correlation** estimates fractional-pixel translation and
emits an inverse warp. See the [subpixel alignment example](examples/pattern_matching/iterative_phase_correlation/README.md).
**Estimate image translation** fits a shift from paired features and
reports inliers and residuals. See the
[matched-feature translation example](examples/pattern_matching/translation_registration/README.md).

The [OpenCV coverage backlog](docs/opencv-coverage.md) ranks missing tools and
links to the full Python binding inventory for the pinned contrib build. The
[roadmap](docs/roadmap.md) puts core vision coverage first and tracks editor,
acquisition and deployment work separately. The [DCT example](examples/image_processing/dct_round_trip/README.md)
displays cosine coefficients and verifies image reconstruction. The
[Fourier filter example](examples/image_processing/fourier_filter/README.md) removes a periodic
stripe pattern with a band-stop filter. The [Fourier gain example](examples/image_processing/fourier_gain/README.md)
suppresses only a chosen pair of frequency bins. The
[Fourier correlation example](examples/pattern_matching/fourier_correlation/README.md)
finds the signed displacement between two images. The [intensity lookup example](examples/image_processing/intensity_lut/README.md)
applies separate intensity curves to the red and blue channels. The
[local statistics example](examples/image_processing/local_statistics/README.md) uses rectangular
neighborhoods to distinguish a flat patch from an equally bright textured patch.
The [intensity transform example](examples/image_processing/intensity_transform/README.md) clips
a ramp and reconstructs it through log and exponential blocks. The
[image arithmetic example](examples/image_processing/image_arithmetic/README.md) subtracts
a blurred background and scales the signed residual. The
[weight-map blend example](examples/image_processing/blend_weight_maps/README.md) feathers
between two aligned images using a per-pixel weight. The
[pixel comparison example](examples/image_processing/pixel_comparison/README.md) turns the
background difference into a binary texture mask. The [nonzero pixels example](examples/image_processing/nonzero_pixels/README.md)
extracts coordinates from that mask and outlines its full bounds. The
[intensity profile example](examples/inspection/intensity_profile/README.md) plots per-column
mean intensity inside an inspection region and locates the maximum in each column. The
[image statistics example](examples/inspection/image_statistics/README.md) reports
selected-region measurements and marks the darkest and brightest pixels. The
[absolute image error example](examples/inspection/absolute_image_error/README.md) compares
average, RMS, and worst-case deviations from a reference image. The
[Earth Mover's Distance example](examples/inspection/earth_movers_distance/README.md)
measures a seven-level intensity shift between histograms. The
[tilted statistics example](examples/image_processing/tilted_statistics/README.md) measures
diagonal patches with 45-degree integral-image windows.

The rotated rectangle overlap example fits a part footprint from a contour
and compares it against a reference. The measurement includes intersection
area, intersection-over-union and the fraction of each footprint covered.
[Rotated rectangle overlap example](examples/geometry/rotated_rectangle_overlap/README.md)

**Convex polygon overlap** measures intersection area and IoU for polygon
footprints. The [example](examples/geometry/convex_overlap/README.md) wires a detected
contour into the block and compares its convex hull with a reference.

**Enclosing convex polygon** fits a footprint under a side-count limit.
The [enclosing polygon example](examples/geometry/enclosing_polygon/README.md) compares
a triangle and a quadrilateral fitted to the same part.

**Delaunay triangulation** connects point sites or measured object centers
and reports indexed triangles, unique edges, and edge lengths.
[Delaunay triangulation example](examples/geometry/delaunay_triangulation/README.md)

**Voronoi cells** divides an image into zones around detected centers or
wired points. It returns bounded polygons, areas, and a mask for a selected zone.
[Voronoi cells example](examples/geometry/voronoi_cells/README.md)

**Minimum spanning tree** finds the lowest total cost to connect nodes
using supplied weights or Euclidean distances from point coordinates.
[Minimum spanning tree example](examples/geometry/minimum_spanning_tree/README.md)

**Path intensity profile** samples each image channel at regular distances
along an open or closed path and preserves gaps outside the selection.
[Path intensity profile example](examples/inspection/path_intensity_profile/README.md)

Convert between Cartesian vector components and magnitude/angle fields.
The angle output carries its units through compatible graph blocks.
[Polar field rotation example](examples/image_processing/polar_fields/README.md)

**Paired Sobel derivatives** exposes signed X and Y gradients for reuse
by **Canny from derivatives**.
[Derivative-fed Canny example](examples/image_processing/canny_derivatives/README.md)

**Scale absolute to 8-bit** applies a gain and offset to signed or high-depth
samples, then produces a clipped magnitude image for display or thresholding.
[Signed-gradient example](examples/image_processing/scale_absolute/README.md)

**Colorize scalar image** maps grayscale, response, or depth values to a
built-in or wired palette. See the [distance-map example](examples/image_processing/colorize_scalar/README.md).

**Labeled distance transform** computes distance to the nearest seed and
returns its region or pixel label. See the [nearest-seed example](examples/image_processing/labeled_distance/README.md).

**Compare pixels** and **In range mask** accept signed label planes. The
[label-mask example](examples/image_processing/label_masks/README.md) selects watershed
boundaries and region interiors.

**Gabor texture response** isolates stripes by orientation and wavelength.
Compare two responses to map the dominant texture direction.
[Gabor texture example](examples/image_processing/gabor_texture/README.md)

**Gabor orientation bank** compares several angles in one block and outputs
the strongest response, winning angle, confidence, and valid-pixel mask.
[Gabor orientation bank example](examples/image_processing/gabor_orientation_bank/README.md)

**Pyramid mean-shift filter** flattens fine color texture before segmentation.
[Mean-shift filtering example](examples/image_processing/mean_shift_filter/README.md)

**Temporal non-local means** uses an ordered group of aligned frames to
denoise a selected frame while retaining its ticket and pixel frame.
[Temporal denoising example](examples/image_processing/temporal_nlm/README.md)

**TV-L1 denoise** restores a monochrome image from one view or a batch of
aligned views with impulse noise.
[TV-L1 denoising example](examples/image_processing/tvl1_denoising/README.md)

**Wiener deblur** restores detail using Gaussian, linear motion, circular
defocus, or custom blur kernels and a configurable noise-to-signal ratio.
[Gaussian example](examples/image_processing/wiener_deblur/README.md) · [Motion example](examples/image_processing/motion_deblur/README.md)

**Adaptive background correction** estimates lighting from a single image
and subtracts or divides out the estimated background.
[Adaptive background example](examples/image_processing/adaptive_background/README.md)

**Hann window** tapers image edges or rows before a Fourier transform and
exposes the coefficients and coherent gain.
[Hann window example](examples/image_processing/hann_window/README.md)

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

Use **Combine regions** to union, intersect, subtract, XOR or invert shapes;
its output is another reusable region, so combinations can be nested. **Measure
selected region** reports pixel area, image coverage, bounds and centroid, with
configurable inspection limits. The [combined regions example](examples/geometry/combined_regions/combined_regions.json) is
an 11-block project that excludes a circle's center and adds a second part region.

**Region from fitted shape** follows a detected ellipse or enclosing circle.
Use its margin to grow or shrink the selection around the fit.
[Fitted ellipse inspection band](examples/geometry/fitted_shape_region/README.md)

**Transform region** carries a selection into another image's coordinates. Connect
the target image, source region and the image operation's `geometry` output, or a
registration `transform`. Forward/inverse direction and mask sampling use dropdowns.
Try [transformed regions](examples/geometry/transformed_regions/transformed_regions.json) for a 12-block crop → resize →
rotation pipeline whose inspection region follows each image operation.

Open [regions](examples/geometry/regions/regions.json) for a working
10-block project that inspects each synthetic part separately, computes a polygon
histogram and blurs only the circular selection. No image download is required.
Run it headlessly with `opencv-dataflow-run examples/geometry/regions/regions.json --repeat 3`.

Selections support filters, segmentation, gradients, image math, histogram and
contrast tools, corner/ORB detectors, contours/components and template search.
Contour/component **Region mode** chooses whole-object center filtering or clipped
measurements. See [region behavior and supported tools](docs/regions.md) for
outside-pixel policies, neighborhood behavior and coordinate rules.

### Blob, line, contour and circle collections

**Find blobs**, **Connected components**, **Find contours**, **Hough line segments**,
**Hough lines from points**, **Hough polar lines**, **Hough circles**, and
**Find line segments (LSD)** expose a `collection`
output. Connect it to
**Filter object collection**, **Sort object collection**, and **Draw object
collection** to select and visualize measured objects without code.

**Hough lines from points** detects lines in sparse coordinates and exposes
the vote count as **Strength**. [Sparse line example](examples/geometry/point_set_lines/README.md)
**Hough polar lines** detects lines in an 8-bit edge map. Vote by pixel count
or intensity and restrict voting with a mask or region.
[Binary and weighted voting example](examples/geometry/hough_polar_lines/README.md)
**Hough circles** exposes fitted radius and native detection strength for sorting
and filtering. [Circle collection example](examples/geometry/hough_circle_collections/README.md)

Use **Clip line segments** to measure only the part of each detected line inside
the image or a rectangular region. [Line clipping example](examples/geometry/clip_line_segments/README.md)

Filter/sort criteria use dropdowns: area, bounding width/height and position,
centroid, aspect ratio, principal-axis angle, line length, perimeter, fitted box
sizes, extent, solidity, circularity, distance and all seven raw/signed-log Hu moments.
Chain filters for multiple criteria, inspect rejected objects, and use stable
two-key sorting with explicit policies for undefined measurements.

Use **Approximate contour** to simplify a detected boundary or open path.
[Contour approximation example](examples/geometry/contour_approximation/README.md)
**Approximate convex contour** reduces a closed boundary to a target number
of sides with an optional added-area budget.
[Convex contour example](examples/geometry/convex_contour_approximation/README.md)
Use **Contour hierarchy** to inspect nested boundaries, holes, and islands.
[Contour hierarchy example](examples/geometry/contour_hierarchy/README.md)
Use **Contour hull and convexity defects** to measure edge notches and judge their count.
[Contour defect inspection example](examples/inspection/contour_defects/README.md)
Use **Contour profile tolerance** to check a measured outline against an aligned reference.
[Contour profile example](examples/inspection/contour_profile/README.md)

Try [object collections](examples/geometry/object_collections/object_collections.json).
This 16-block project orders two parts left-to-right, selects the round contour and
isolates two long vertical lines. See [collection tools and measurement conventions](docs/object-collections.md).

### Channels, payloads and image pyramids

Use **Split three channels / Merge three channels**, or the four-channel alpha
variants, to process planes independently. **Extract image channel** and **Replace
image channel** use channel-index dropdowns and preserve coordinate payloads.
**Route image channels** assembles channels from one or two images with a
configurable or wired index map. See the [false-color example](examples/image_processing/route_channels/README.md).

**Gaussian image pyramid set** sends all levels together on its `images` port.
Connect it to an ordinary image input: the largest level is used by default.
Choose a smaller level using the inspector's **Pyramid level: input name** dropdown.
**Select pyramid level** provides an explicit single-image output; **Pyramid down**
and **Pyramid up** are separate toolbox entries too.

Images and other port values can carry optional JSON payloads. Built-in spatial
payloads record image dimensions, the mapping to the original image and pyramid
level provenance. Results stay in their working coordinates and show the source
level in the inspector and headless receipts. **Read port payload / coordinates**
exposes a mapping to the largest or original image for **Transform points** and
**Transform region**. **Attach port payload** adds recipe or custom data.

Try [channels and pyramids](examples/image_processing/channels_pyramids/channels_pyramids.json): it splits/merges channels, carries a recipe
payload through a four-level pyramid, detects blobs at level 2, and maps their
centroids back to the largest image. See [payload and pyramid contracts](docs/payloads-pyramids.md).

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
ordinary Grayscale supports uint8/uint16/float32. Canny and Adaptive Threshold
accept color or grayscale and use an 8-bit working image for their masks. Equalize
Histogram, CLAHE, and Inpaint can preserve the source pixel depth and color encoding.
Legacy G-API Grayscale also requires uint8. For explicit conversion, choose the
color format and pixel depth in the file input or a Convert image block.

### Video file replay and recording

**Video file** emits one decoded frame per graph activation. Set a zero-based
start frame, optional exclusive end frame, stride and stop or loop mode.
The frame ticket and metadata include its filename and index. The metadata
also gives decoder time or an explicitly labeled frame-rate estimate.

Connect the result to **Write video file** to encode processed frames at a
configured playback rate. Choose a new output path and a codec supported by
the selected backend. A continuous run or a headless run with `--all-images`
writes one clip, which is finalized when execution closes.
See [video file workflows](docs/video-files.md).

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
[`examples/user_tools/inspect_shapes/inspect_shapes.toolblock.json`](examples/user_tools/inspect_shapes/inspect_shapes.toolblock.json)
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

**Sequence** blocks retain memory for running background subtraction, Farneback optical
flow, Lucas-Kanade point tracking and first/latest/mean image references. They report
readiness and activation counts, suppress primary outputs during warmup, and support
bounded per-stream state, resets and trained references stored in projects or recipes.
Try [sequence memory](examples/flow_control/sequence_memory/sequence_memory.json), then run continuously. See
[sequence memory](docs/sequence-memory.md) for training, masks, ticket provenance and limits.

**Fit feature covariance** models how numeric measurements vary together.
**Mahalanobis distance** scores new rows and can fail on an outlier.
[Feature covariance example](examples/numerical/feature_covariance/README.md)

**Solve linear system** fits one or more observation columns and reports residuals
and numerical rank. See the [linear system example](examples/numerical/linear_system/README.md).

**Solve linear program** optimizes continuous allocations under linear limits.
The [linear program example](examples/numerical/linear_program/README.md) shows capacity
allocation and minimum-cost demand fulfillment.

**Solve polynomial roots** finds curve crossings or complex roots and checks each
candidate's relative residual. See the [polynomial roots example](examples/numerical/polynomial_roots/README.md).

**Multiply matrices** applies weighted matrix products and optional offsets.
The [matrix multiplication example](examples/numerical/matrix_multiply/README.md) uses
fitted coefficients to predict new readings.

**Invert matrix** computes inverses or SVD pseudoinverses and reports rank,
conditioning, and square-matrix determinants. See the [matrix inverse example](examples/numerical/matrix_inverse/README.md).

**Singular value decomposition** exposes the spectrum, vector bases, numerical
rank, and an optional low-rank reconstruction. See the [SVD example](examples/numerical/matrix_svd/README.md).

**Eigendecomposition** extracts real eigenpairs from square matrices and
checks their residuals. See the [eigendecomposition example](examples/numerical/matrix_eigen/README.md).

**Check image samples** detects nonfinite and optionally out-of-range pixels.
It emits aligned validity masks for downstream analysis. See the
[image validity example](examples/image_processing/image_validity/README.md).
**Replace nonfinite samples** substitutes configured values for NaN and
infinity and emits a mask of changed pixels. See the
[nonfinite replacement example](examples/image_processing/nonfinite_replacement/README.md).

**Join images** assembles a side-by-side or stacked view without resampling.
Its layout output records each panel's source frame. See the
[image join example](examples/image_processing/join_images/README.md).
**Paste image patch** copies selected pixels onto a canvas and reports
their placement. See the [masked paste example](examples/image_processing/paste_patch/README.md).
**Crop subpixel patch** samples around a fractional point or pose origin
and carries the patch-to-source transform. See the
[subpixel crop example](examples/pattern_matching/subpixel_crop/README.md).
**Polar image warp** straightens circular features into a radius-by-angle grid
and carries a mapping for rewrapping the result. See the
[circular inspection example](examples/image_processing/polar_image_warp/README.md).
**Tile image** repeats an image in a row-and-column grid. The
[tiling example](examples/image_processing/tile_image/README.md) uses the grid as a synthetic
fixture for template matching.
**Transpose image** swaps rows and columns without interpolation. The
[transpose example](examples/image_processing/transpose_image/README.md) maps points from
the transposed crop back to the original image.

**Gram matrix** compares rows or columns by dot product and can reuse the same
centering offset across batches. See the [Gram matrix example](examples/numerical/gram_matrix/README.md).

**Find nearest rows** compares numeric feature tables and returns exact nearest
reference indices and distances. See the [nearest rows example](examples/numerical/nearest_rows/README.md).

**Sort numeric table** ranks records or sorts values within each row or column
and returns their source indices. See the [numeric table sort example](examples/numerical/numeric_table_sort/README.md).

**Fit principal components** reduces numeric feature tables by count or
retained variance. **Transform principal components** projects rows or
reconstructs them with the same fitted model.
[Principal components example](examples/ml/principal_components/README.md)

**Measure image norm** computes a signal magnitude or reference difference inside a
mask or region. Choose infinity, L1, L2, squared L2, Hamming or Hamming2 and set an
optional upper limit. Try the [image norms example](examples/inspection/image_norm/README.md).

**Combine Fourier spectra** multiplies two typed DFT outputs or conjugates a spectrum.
The product can drive a custom frequency-domain filter or a circular convolution.
Multiplying by a conjugate gives a separate circular lag frame after IDFT.
Try the [spectrum product example](examples/image_processing/spectrum_product/README.md).

**Generate random image** and **Add image noise** provide repeatable Gaussian and
uniform fixtures with saved or wired seeds. Add noise inside a mask or region
while keeping the original encoding and depth.
Try the [seeded noise example](examples/image_processing/seeded_noise/README.md).

**Create diagonal matrix**, **Complete symmetric matrix**, and **Measure matrix diagonal**
prepare and inspect numeric tables for fitting, regularization, and inversion.
Try the [matrix structure example](examples/numerical/matrix_structure/README.md).

**Stitch image pair** combines overlapping inspection views in a shared frame
or with a wired registration. It reports coverage, overlap, and the mappings
from both inputs. Try the [stitching example](examples/image_processing/stitch_image_pair/README.md).

**Stabilize camera motion** tracks stable background features to reduce
translation or similarity jitter before measuring a frame. It publishes
corrected pixels, a coverage mask, and the mapping back to the raw image.
Try the [camera stabilization example](examples/flow_control/camera_stabilization/README.md).

**Stitch scan sequence** automatically registers two to eight ordered views.
Each adjacent pair must meet the fit and overlap thresholds before the graph
receives a mosaic. Try the [ordered scan example](examples/image_processing/stitch_scan_sequence/README.md).
For scans with uneven lighting, the [exposure example](examples/image_processing/stitch_scan_exposure/README.md)
shows optional gain correction and feathered blending.

## Architecture

Choose **Run profile: debug / balanced / lean** in the toolbar to control observation
overhead. Lean skips routine diagnostic capture and image preview transfers, samples GUI
updates, and removes the development pacing delay while preserving processing outputs,
failure reasons, counters and private input isolation. The headless runner accepts
`--profile lean`. See [run profiles and measured performance](docs/run-profiles.md).

Use **Graphics default** in the toolbar to turn optional result graphics on or off
across the project, or set **Result graphics** on a block to override the default.
Declared image outputs remain available. Both choices are saved with the project.

Independent graph branches can run concurrently as soon as their inputs are ready.
Use **Branch workers** in the toolbar to save **Auto**, **Serial**, or a worker
limit for the current graph. The headless runner accepts `--branch-workers 1` to
run the root graph serially. See [branch scheduling](docs/control-flow.md).

The bottom status bar shows running ticket totals, pass/fail counts, pass rate and
average end-to-end time. Final judgments can attach stable failure codes and explanations.
Use **Results > Validation image sets…** to grade saved images as must/should pass,
unknown or must/should fail, optionally matching a preferred failure code. Runs produce
a score and per-image report without affecting production counts. Try [validation image set](examples/inspection/validation_images/validation_images.json); see [production counts and validation](docs/production-validation.md).

Open **Results > Block diagnostics** (**Ctrl+Shift+D**) for a bottom dock with
severity-colored logs. Selecting a block isolates its logs; clearing the selection shows
all blocks. Filter verbosity/severity, search messages, and inspect structured parameters
and chained exception tracebacks. Every block exposes `block.logger` for custom
diagnostics. See [block diagnostics](docs/diagnostics.md).

**Results > Ticket activity timeline…** displays a ticket Gantt chart, while
**Results > Block concurrency…** shows verdict-colored activity bars in a rolling
time window. Both use a common block start/finish event log with ticket identities,
exception/blocked-input/queue flags, filters, pause, zoom and JSON export. Nested and
parallel inspections share the same collector. See [activity events and timelines](docs/activity-timelines.md).

Every block retains current/last/average execution timing and an activation count.
The inspector shows these statistics, and headless reports include them. Ticketed
connections carry source-start/output-finish UTC timestamps and monotonic elapsed time,
including queue waits. **Results > Ticket results…** shows the overall observed span
and per-tool durations. See [execution and ticket timing](docs/timing.md) for measurement
boundaries, parallel work and explicit final-decision timestamps.

- `core.py`: block/result contracts, typed ports, acyclic graph scheduling, JSON projects.
- `blocks.py`: acquisition, legacy G-API processing, inspection, and output sinks.
- `basic_processing.py`: ordinary Grayscale, Gaussian blur and Canny defaults.
- `processing.py`: base OpenCV geometry, filtering, segmentation, math and inspection wrappers.
- `features.py`, `registration.py`, `analysis_tools.py`: contrast, features, registration and numerical image tools.
- `dnn_blocks.py`, `lightglue_blocks.py`, `annoy_blocks.py`: inference, learned matching and vector search.
- `geometry3d.py`, `stereo3d.py`, `ptcloud3d.py`: calibrated geometry, depth, motion and fusion.
- `pointclouds.py`, `cloud_tools.py`, `cloud_io.py`, `cloud_viewer.py`: typed clouds, surface matching, files and viewing.
- `regions.py`, `selection.py`: reusable shape regions, mask intersections and selection policies.
- `region_editor.py`: interactive image-space drawing and editing of region shapes.
- `payloads.py`, `pyramids.py`: optional wire payloads, image coordinate provenance and bundled pyramids.
- `objects.py`, `collection_tools.py`: measured blob/line/contour/circle collections, filtering, sorting and rendering.
- `examples.py` and `runner.py`: built-in starter graphs and headless project verification.
- `plugins.py`: explicitly loaded trusted Python extensions.
- `plugin_dialog.py`: plugin editor context and cached image snapshots.
- `canvas.py`: movable nodes, typed sockets, selectable wires, pan and zoom.
- `app.py`: Qt editor, inspector, worker execution, project and plugin actions.
- `toolblocks.py`: composite blocks, public terminal mappings, reusable definitions.
- `toolblock_editor.py`: public terminal configuration dialog.
- [`examples/`](examples/README.md): runnable projects, reusable ToolBlocks, and a Python plugin,
  each with its own README.
- `tests/`: local runtime and persistence checks.

Default Grayscale, Gaussian blur and Canny use ordinary OpenCV calls, retaining
per-node results for inspection. Legacy variants under **Advanced > G-API** each
cache a `cv2.GComputation`; the editor graph is not fused. An explicitly compiled
ToolBlock with selected diagnostics is [future work](docs/opencv-backend.md#compiled-toolblocks-future),
subject to demonstrated performance benefit before enablement or expansion.
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
