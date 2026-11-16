# SimpleLaserProfile training and inspection

Train a single laser stripe from an image, then extract source-pixel centers
and, with known camera/laser calibration, measured 3D profile points. Existing
training reopens with its original image, region, settings and stripe overlay.

## Open the tool

In OpenCVDataflowGUI choose **Load Python plugin** and select this folder's
`plugin.py`. It explicitly loads the pinned **dev2** wheel from `releases`;
no host environment installation or model download is required. Restart the app
before switching from a previously loaded version. The tool is
`3d.simple_laser_profile` / **SimpleLaserProfile**.

Then open an execution project:

| Project | Expected availability result at the same 90% threshold |
| --- | --- |
| [Generated calibrated step](execution/calibrated_step/project.json) | Pass: 480/480 bins, measured 50-mm step ≈49.998 mm. Software control, not a real gauge qualification. |
| [Generated missing/competing/input-ceiling case](execution/calibrated_unavailable/project.json) | Fail: 390/480; the 90 rejected bins stay out of the cloud. |
| [Real WLI reference](execution/wli_1/project.json) | Fail: 72/120; input-ceiling rejection remains visible. |
| [Real WLI second capture](execution/wli_2/project.json) | Fail: 42/120; missing and ambiguous bins remain visible. |
| [Real WLI poor fit 3](execution/wli_3/project.json) | Fail: 2/120; clipping and poor reference agreement retained. |
| [Real WLI poor fit 4](execution/wli_4/project.json) | Fail: 7/120; clipping and poor reference agreement retained. |

These projects execute the algorithm. The frozen recipe/settings and numerical
reports match the standalone studies. An availability pass is not stripe-identity,
dimensional or gauge acceptance. Failure results keep their raw measured evidence.
The same gate is used for all six projects; no per-image threshold relearning.

## Train a new application

1. Add the tool, connect an original image and save the project. Choose
   **Train laser profile…** in its Configuration inspector.
2. **Import reference** or **Capture connected reference**. A connected capture
   reads an already published image; it does not run the graph. Preview another
   image separately to avoid replacing training by accident.
3. Select one stripe with **Draw new rectangle**. Move/resize the handles, or
   enter source X/Y/width/height and choose **Set rectangle**. Scroll/zoom, Fit,
   Pan and 100% help with fine edits. Import a binary mask for a nonrectangular
   region; it remains exact until deliberately replaced.
4. Optionally import a matching laser-off background. Import known camera/laser
   calibration for physical output, or explicitly use **pixel-only** mode.
   The [generated calibration definition](studies/calibrated_control/calibration.json)
   documents the schema; it is not a calibration for a real camera.
5. **Learn / preview reference** learns direction, contrast threshold and width
   once. **Preview another image** or **Preview connected image** runs those
   frozen settings. Review coverage, rejected candidates, chart and native cloud.
6. **Stage recipe**, then **Apply configuration** in the host. Stage writes a
   new immutable `.slp` asset; it never overwrites the previous recipe. Cancel
   before Stage leaves settings/files unchanged. Canceling the host draft after
   Stage can leave an unused asset while preserving the applied recipe.

Reopening restores the retained image, ROI/mask, optional background, camera/
laser calibration and settings, and displays the trained stripe. Editing clears
stale results and requires learning again. Separate test previews never train.
Workers stop publishing when canceled, edited, closed or superseded by a newer
connected preview. Save As copies the sealed recipe with its hash.

The first supported scope imports known pinhole-camera/laser-plane calibration.
Automatic camera/board/plane calibration and motion/encoder scan accumulation
remain follow-ups. Original dimensions, channels and encoding must match training;
cropped, warped and smaller pyramid grids require corresponding calibration and
are rejected by this initial integration.

## Assess results

Calibrated examples default to the native **fly-around cloud thumbnail/popout**.
Double-click the cloud and enable Flyaround to orbit it. Points use meters,
the camera frame and RGB colors; missing samples and hidden surfaces are not
filled. Cloud pixel indices are rounded source pixels; the report keeps the
original subpixel centers.

Choose the tool's **overlay** or **chart** result to inspect its images. The
unchanged **image** result also carries drawable stripe graphics. Enable the
host's result-thumbnail capture when using the image results filmstrip.

- Green: accepted centers. Magenta: competing peaks. Red: input encoding ceiling.
- Amber: clipping/width rejection. Cyan: selected region. Blue border ticks: missing bins.
- The chart retains gaps and has a source-bin availability strip.
- `report` retains every bin/candidate/reason and XYZ in meters when calibrated.
- `coverage` is valid/active bins, not accuracy or confidence. `count` is accepted bins.

Graphics off preserves numerical points/counts/decisions and skips colors,
drawings, overlays and plots. Pixel-only or empty metric results emit **no cloud**.

## Earlier saved-result reviews and attribution

The eight [saved-result review projects](studies/host_review_checks.json) remain
under `studies`. Open a `studies/.../project.json` without loading a plugin to
review retained images and clouds. Their file-source PASS means loading only;
they do not execute extraction. The original dev1 wheel remains for that snapshot.

WLI-Set v3 is real **CC BY 4.0** data by **zeng song**, credited in
[DATASET_CREDITS.md](DATASET_CREDITS.md), with byte-exact originals and published
software centerline masks. Those captures have no camera/laser-plane calibration
and remain pixel-only. The poor fits are retained. Nearest software-centerline
distance does not establish stripe identity, completeness or physical accuracy.
Original tool code/generated controls use [MIT](LICENSE). Dependencies retain
their NumPy/OpenCV/Pillow/PySide6 licenses. No ML runtime or weights are used.

For the standalone API/CLI, install the optional dev2 wheel in a separate tool
environment and use `simple-laser-profile train`, `inspect` and `run`.
