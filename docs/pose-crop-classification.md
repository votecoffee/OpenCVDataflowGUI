# Pose-aligned crops and per-part classification

**Crop at image pose** (`cv.crop_pose`) is under OpenCV → Image Processing →
Geometry. It samples a fixed local inspection window using a measured image pose.
Use it inside the existing parallel inspection ToolBlock with a classifier or any
other crop-based inspection. The [eleven projects](<../examples/opencv tools/rotated_crop_classification/README.md>)
show that composition without adding a classifier-specific scheduler.

## Pose and pixel coordinates

Connect an intensity image and an affine `image_pose_v1`. The pose origin is the
crop center unless local Center X/Y offsets are configured. Local Width/Height
are pixel-cell spans in pose units; Width/Height are output pixel counts. Pose
rotation, scale, unequal axes and shear remain intact. A full-resolution pose can
be used on a smaller pyramid grid when both have a known shared coordinate root.
Projective poses require the existing Rectify quadrilateral tool instead.

For output column x, the local coordinate is:
`center_x + (x + 0.5) * local_width / width - local_width / 2`.
Rows follow the same half-pixel convention. The output-to-input map composes this
local sampling with the pose mapped into the current image. Native warpAffine
uses that inverse map; no fitted axis-aligned replacement box is introduced.

Geometry includes input/output sizes, output-to-input, output-to-pose,
pose-to-input, source cell-footprint corners and measured coverage. Crop pixels,
valid mask and geometry carry moved spatial/pyramid mappings. Image encoding and
stored uint8/uint16/int16/float units remain unchanged by sampling. Depth-channel
physical planes use their dedicated tools. The source image is not modified.

## Support, borders and outcomes

Masks and regions intersect in the source grid, and zero-alpha pixels are
excluded. Excluded samples are replaced with Border value before interpolation.
The fill uses stored sample units and must fit an integer source depth. Wired
optional selections require current values rather than falling back to a whole
image. Source/crop pixel budgets and bounded coordinates are checked before
sampling allocations.

The valid mask marks measured samples, independent of border extrapolation.
Nearest/linear support follows positive native interpolation weights. Cubic and
Lanczos use a conservative full-kernel halo so negative lobes cannot falsely
claim support. The coverage output is the fraction of fully supported crop pixels.

Strict mode rejects any unsupported sample and emits no current result. Pad mode
allows constant/replicated/reflected/wrapped image-edge values, while those values
remain unsupported. Minimum coverage produces a normal inspection judgment with
`crop.coverage` reason; it does not silently make padded pixels measured. Downstream
inputs can reject failed crops using the standard failure policy. The crop's
score describes coverage, independently of a classifier's score.

Drawable source footprints follow the graphics policy. Annotation-only steps emit
a completed empty graphics layer when generation is disabled; a Lean run therefore
does not mistake that completed diagnostic step for unfinished buffered work.
No shared scheduler semantics or source acquisition policy was changed.

## Tickets and reusable inspection

Known image/pose tickets must agree. In a per-part job the image has a derived
child ticket, while an embedded measured pose can retain its known parent ticket;
that explicit inspection relationship is accepted. Unrelated tickets or coordinate
roots are rejected. Missing identity is not proof of physical synchronization.

The example ToolBlock composes existing tools:

1. Accept source image and pose; sample a 64×32 local crop.
2. Convert the inference copy to gray uint8, preserving the public crop's depth.
3. Run a sealed 32×16 native ONNX classifier and the existing class decoder.
4. Expose class, confidence, crop geometry, crop pixels and an optional full-image
   label/footprint overlay.

`flow.inspect_each` clones this graph for each measured blob pose, with bounded
workers and aggregate pass/fail after the jobs settle. Each record retains the
feature ID and source map. The images group retains child tickets and an
`inspection.parent_ticket` reference. Serial and parallel versions use the same
ToolBlock; no runtime plugin or custom execution code is required. Parent overlays
show per-part status/IDs; enter the inspection graph for the selected part's class
label, source overlay and intermediate results.

## Original controlled model and evidence

The original synthetic parts have rectangular silhouettes and horizontal/vertical
local stripe textures. Their thresholded silhouettes give measured blob moments,
centers and an unoriented 180-degree axis. Both classes are unchanged by that
ambiguity; the fixture makes no claim of directed front/back orientation.

Twenty-four noisy canonical windows train an OpenCV binary linear SVM. Its signed
weights and bias are exported as a small Flatten/Gemm ONNX model; logits are the
two signed margins times three. Training and inference both resize in float32
before 1/255 normalization. Scaling those margins does not calibrate confidence.
The bundled model is 4,313 bytes with a sealed configuration and original training
windows/coefficients. Scene noise seeds differ from training; the images are drawn
analytically rather than by the crop sampler being tested.

Five marked parts pass the illustrative 0.9 threshold (minimum score about 0.9966).
An unmarked part has score about 0.5014 and fails. The same scene without pose
alignment has zero passes; an all-known scene has six. Horizontal-only acceptance
has three passes/three failures. These related synthetic controls demonstrate
integration, not production accuracy or reliable unknown-class detection.

An independent reference measures component moments directly, derives coordinate
maps, evaluates bilinear pixels in NumPy and compares native ONNX logits with
exported coefficients. Maximum pixel disagreement is below 0.501 uint8 code units
and logit disagreement below 5.4e-7. Debug/Lean, high-depth inputs, selections,
coverage failures, exact regeneration, nested JSON/XML/Save As, failed-input gating
and cold worker teardown are checked. No accelerator, equipment, third-party
training data or inference-speed benefit is claimed.
