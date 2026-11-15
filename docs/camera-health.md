# Monitor camera health

**Monitor camera health** (`inspect.camera_health`) is a Custom inspection tool
for a normally fixed camera. It compares each image with a SHA-256-pinned saved
reference. It monitors background/fiducial position, lighting and focus, with
separate warning and alarm limits. It does not silently learn camera drift or
replace the baseline while inspecting.

## Train and run

1. Add the block and connect the camera/file image. Save the project. Run the
   source to publish an image; an untrained monitor reports that training is needed.
2. Open **Train camera reference…** in its configuration sidebar. Capture the
   wired input or import one or more known-good images. The modeless trainer can
   stay open while acquisitions continue; each capture copies a completed image
   publication and does not execute the graph.
3. In **Regions / fiducials**, draw named rectangle, circle or polygon regions.
   Add Lighting and Focus regions on static, representative areas. Add Ignore
   regions over expected part paths, Position regions to restrict background
   tracking, or Fiducials over unique static visual landmarks.
4. Choose Background or Fiducials position checking. Set the **Warning / alarm
   limits** tab, or edit the corresponding ordinary block parameters later.
5. Click **Train reference**, inspect historical means/standard deviations, then
   **Stage reference / limits**. Click **Apply configuration** in the inspector
   to activate the staged model. Canceling the dialog does not alter the block.
6. Run normally. Inspect State, Report, foreground/background masks and drawable
   colored region outlines. Boolean Warning/Alarm outputs can feed decisions or
   reporting blocks.

Reference captures must use consistent resolution, color format, depth and pixel
grid. One capture provides a baseline; several captures allow the median image
and temporal range mask to suppress regions that changed during training.
Training does not prove that captures came from separate physical exposures.
The reference archive retains normalized intensity history and per-region sample
values/statistics, allowing later review/retraining. Alpha is excluded from health
measurements; reopened training history reconstructs alpha as opaque.

## Position and foreground

Background mode tracks stable Shi-Tomasi corners using pyramidal Lucas–Kanade,
checks forward/backward agreement and fits a RANSAC similarity transform. Local
contrast preprocessing affects only tracking, not reported photometry or focus.
After a rough fit it identifies coherent appearance changes, removes foreground
tracks and refits the background. Minimum track count, inlier fraction and spatial
spread prevent a small unsupported point cluster from being called healthy.

Automatic foreground detection compensates approximate channel gain/offset before
thresholding changes; opening, component area and dilation remove isolated noise
and enlarge part boundaries. Its thresholds are configurable. A connected
**Foreground mask** can explicitly mark expected current-image movers/occlusions.
Ignore regions and unstable training pixels provide additional fixed exclusions.
Foreground/background masks are exposed for inspection and reuse.

Fiducial mode uses masked normalized correlation in a bounded search around each
trained region. Every trained landmark must exceed the score and competing-peak
limits. Missing/occluded/ambiguous landmarks alarm. A single fiducial measures
translation only: rotation and scale are explicitly unmeasured. Two or more
consistent landmark centers also estimate in-plane rotation and scale.

Position displacement is measured at the reference image center in pixels.
Separate warning/alarm pairs cover that displacement, absolute rotation in
degrees and absolute scale change in percent. Transforms map reference pixels
into current pixels; they are image-plane estimates, not robot/extrinsic poses.

The scene must provide enough visible fixed structure. Large/repetitive movers,
major occlusion, deformation, extreme lighting or motion outside the configured
search/tracking range can make position unavailable. That is an alarm with a
reason, not a healthy zero-motion measurement. Use known static position regions
or fiducials when the scene has little usable background.

## Lighting and focus regions

Each named region is evaluated independently. Region masks follow a measured
position transform into current coordinates. Scores use native current pixels;
focus images are not resampled into the reference and blurred by that warp.
The corresponding visible reference mask supplies a matched comparison population.
Unstable, foreground, occluded and out-of-frame pixels are excluded. Insufficient
visible coverage alarms. If position is unavailable, region checks use fixed
pixel coordinates and explicitly report that position compensation was unavailable;
the overall result still alarms for the missing position evidence.

Lighting reports each intensity channel's current/reference mean and signed
percentage change, plus the largest absolute channel change. A configurable
denominator floor handles near-black regions. Historical full-region capture
statistics remain in the report; matched visible-reference means are also given.
Integer samples normalize to their full type range; floating samples use configured
Float min/max. Changing normalization settings requires retraining.

Focus reports Tenengrad or Laplacian-variance sharpness and percentage loss from
the saved reference over the corresponding visible selection. Optional variance
normalization reduces gain/offset sensitivity. Keep metric, aperture and
normalization fixed after training; changes require retraining. Flat or inadequately
visible regions alarm because they cannot establish focus health. Sharpening gains
are not classified as focus loss. These are image-based indicators, not a physical
lens/focus calibration.

Localized lighting changes can resemble foreground; broad camera/scene changes
can overwhelm tracking. Inspect masks and coverage, choose stable regions and use
explicit mover masks/exclusions where needed. Disabling automatic foreground is
available when those selections already describe the scene reliably.

## States, persistence and limits

- **Healthy:** every enabled check has sufficient evidence and stays below warning.
- **Warning:** at least one warning limit is reached and no alarm occurs. The tool
  still passes; Warning is true and overlays are amber.
- **Alarm:** an alarm limit is reached or enabled checks lack evidence. The tool
  fails with per-check reasons; Alarm is true and overlays are red.

Thresholds trigger at or above their limits; warning must be strictly below alarm.
The report contains per-region values, historical statistics, fitted position,
match/track quality, exclusions, reference identity and active limits. Images and
results retain the current ticket, recipe and image coordinates. Normal block
timing, diagnostics, failure-input policy and exception containment apply.
Graphics-off skips drawable outlines without changing judgments. Training and
runtime operation do not acquire devices or write PLC/robot values.

References are project-contained, immutable revision assets with pinned hashes.
Save As copies them without changing bytes; project/XML settings retain the path
and hash. A changed asset, input layout or incompatible trained setting is rejected.
The archive is pickle-free and bounded. Each image is limited to 2,097,152 pixels
and 16,384 pixels per axis; history to 32 captures and 32 million intensity samples
in total. Large color captures therefore allow fewer history frames. Training runs
on private copies in a background worker. No reference update occurs without
Train, Stage and Apply.
