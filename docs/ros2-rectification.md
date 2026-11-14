# Rectify a ROS intensity image

**Rectify ROS 2 intensity image** (`industrial.ros2_rectify_image`) explicitly
warps raw pixels using CameraInfo K/D, R and the left 3×3 portion of P. It supports
monocular pinhole and fisheye calibration, including nonidentity R, raw sensor ROI
and binning, and explicit rectified output grids.
The existing CameraInfo decoder remains a converter for already selected image
space; it does not perform this warp.

The geometry follows the [ROS CameraInfo definition](https://github.com/ros2/common_interfaces/blob/lyrical/sensor_msgs/msg/CameraInfo.msg)
and OpenCV's [camera model](https://github.com/opencv/opencv/blob/5.x/modules/calib/include/opencv2/calib.hpp).
Pinhole and fisheye paths use their respective `initUndistortRectifyMap` functions,
then `remap`. This block needs OpenCV but no ROS interpreter or middleware.

## Inputs and configuration

Connect a raw intensity Image and either complete CameraInfo fields or a decoded
**Raw** CameraInfo camera model. A model from **Cache ROS 2 CameraInfo** is allowed
only with its matching image ticket and pixel grid. An already Rectified model
is rejected to avoid using its original raw distortion again. If CameraInfo is
unwired, the Camera info JSON parameter supplies a saved fallback.

CameraInfo width/height describe the full calibrated sensor. Its raw ROI is in
full-resolution pixels; zero binning means one and all-zero ROI means the full
sensor. Source dimensions must equal the complete ROI bins. Raw K is adjusted
by subtracting the raw ROI offset, then dividing by binning independently on each
axis. Image dimensions, known ROS dimensions, optical header frame and pixel
coordinates are checked. The received grid may have identity pixel coordinates,
or an exact declared crop/binning transform from the full sensor. Unrelated
software warps are rejected. R must be a proper rotation and P must have zero
translation; stereo-baseline P still requires separate geometry handling.

Choose the **rectified** output grid separately from the raw acquisition ROI.
**Camera info ROI** uses the same numeric ROI only when `roi.do_rectify` is false.
When it is true, choose **Full sensor** or **Custom ROI** explicitly. This block
does not estimate a tight rectified ROI. Full sensor selects the calibrated
sensor canvas; Custom ROI selects a rectangle in full-resolution rectified P
coordinates. Either may contain pixels outside the acquired raw region: inspect
Valid mask before downstream measurements. Full sensor does not recover uncaptured
pixels. Custom ROI must fit the full sensor canvas.

Supported samples are gray uint8/uint16/int16/float32/float64 and uint8/uint16
RGB/BGR(A). Format and sample depth are preserved; nonnative byte order is
normalized before OpenCV sampling. **Scalar semantics → Auto** rejects ambiguous
floating-point or `16UC1` planes. Choose **Intensity** only
when these values represent brightness or discrete labels. Physical `DepthArray`
input is always rejected: rotating axial depth requires 3D reprojection, not
ordinary image sampling. Untagged data still needs the correct caller-declared
meaning. Use Nearest interpolation for discrete labels.

| Control | Behavior |
| --- | --- |
| Rectified frame | Empty derives `<source>_rectified`; nonidentity R requires a distinct frame name |
| Rectified grid | Camera info ROI, full sensor canvas or custom rectified ROI |
| Rectified ROI | `[x,y,width,height]` in full-resolution rectified pixels, used only for Custom ROI |
| Width / Height | Both zero keep the selected ROI divided by source binning; both positive resize that grid and scale its intrinsics |
| Interpolation | Nearest, linear, cubic or Lanczos4 |
| Border / Border value | OpenCV constant, replicate or reflect choices and finite channel values in the existing channel order |
| Header mode | Require an image acquisition header, or explicitly use zero if it is absent |
| Require matching ticket | Check wired calibration against the image acquisition ticket; a statically bound model is checked regardless |

Output is bounded to 4,194,304 pixels and 16 MiB. A new calibration or output
geometry rebuilds maps; changing an acquisition stamp does not. Runtime map
buffers are private and are not saved or shared between execution snapshots.

For output intrinsics, subtract the selected **rectified** ROI offset from P's
principal point, divide the first and second rows by their binning factors, then
scale those rows for an optional output resize. Incomplete edge bins are dropped
and reported for both source and output. This follows the CameraInfo sampling
convention without adding a half-pixel shift. Info records both source geometry
and `rectified_output_to_full`; the latter maps output pixels to the chosen
virtual full-resolution rectified canvas, not back to distorted raw pixels.

## Results and provenance

- **Image** contains remapped samples on a new pixel frame. Pyramid links and
  old static/RGB-D bindings are removed. A nonlinear warp is not represented as
  an affine transform back to the raw image.
- **Camera** and **Camera info** describe the new grid with P-derived/scaled
  intrinsics, zero distortion, identity R and zero P translation. Their header
  uses the image's acquisition stamp and new optical frame. The original
  calibration timestamp is never substituted for missing image time.
- **Pose** maps raw optical vectors **into the rectified optical frame**:
  `p_rectified = R @ p_raw`, with zero translation. This virtual frame rotation
  is not a measured robot/camera mount transform. The pose encoder can invert it
  when the opposite parent/child convention is required.
- **Valid mask** is uint8 source-center coverage in front of the camera and inside
  the image, with a 1e-4 pixel boundary tolerance for map roundoff. It does not
  certify full cubic/Lanczos kernel support, scene visibility or pixel quality.
  Extrapolated borders stay invalid; back-facing rays are filled with Border value.
- **Map X/Y** are destination-to-source float maps, hidden as connectors by
  default. Their coordinate tags interoperate with generic Remap's pixel-frame
  checks. Generic Remap alone does not update ROS optical headers/CameraInfo.
- **Info** retains original CameraInfo, image header/pixel frame, optional source
  calibration ticket/binding, geometry signature, source/output ROI and binning,
  discarded edge bins and coverage measurements. `rectified_roi_estimated` is false.

Each output keeps the image ticket and applicable recipe/run context. Tagged
calibration from a different run generation and inconsistent managed recipes
are rejected. The coverage score is valid pixels / output pixels. A wholly
unsupported view emits a failed result with `no_valid_rectified_pixels` rather
than an exception. Invalid geometry/settings emit no result; the host contains
unexpected native exceptions through its normal block execution wrapper.

## Examples and checks

Open [the PCB software fixture](<../examples/industrial integration/ros2_fastdds/offline_intensity_rectification.json>)
or [the cropped/binned PCB fixture](<../examples/industrial integration/ros2_fastdds/offline_roi_binning_rectification.json>),
or [the disabled raw-image subscriber](<../examples/industrial integration/ros2_fastdds/intensity_rectification_subscriber.json>).
The PCB photo is openly licensed, but **its calibration parameters are synthetic**
and image acquisition time is explicitly unknown/zero. It demonstrates the
workflow without claiming real lens correction or measurement accuracy. The ROI
fixture also uses synthetic acquisition geometry: a software crop/decimation is
not a hardware binning model. Its 160×120 source is rectified onto an explicitly
selected 240×160 canvas; invalid borders remain visible in Valid mask.

The live template primes static raw CameraInfo, binds each raw intensity image,
then rectifies it. It starts disabled and launches no simulator or robot.
Focused checks cover native pinhole/fisheye maps and pixels, independent point
projection, high-depth/color formats, headers, frame separation, mask/map reuse,
ticket/recipe/run isolation, failures, XML and offscreen editor controls:

```powershell
python -m pytest tests/test_ros2_rectification.py tests/test_ros2_rectification_roi.py -q
```

Stereo baselines, automatic tight rectified ROI estimation and physically correct depth
rectification remain separate milestones. Existing stereo tools retain their
own explicit calibrated-rig contracts.
