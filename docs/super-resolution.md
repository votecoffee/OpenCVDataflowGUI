# Learned image super-resolution

`ml.super_resolution` runs OpenCV's single-image `dnn_superres` interface and
returns a learned image, an interpolation baseline, a processed-selection mask,
nominal geometry and execution information. Start with the
[eight saved projects](<../examples/opencv tools/super_resolution/README.md>).
They include unchanged, hash-pinned Apache-2.0 FSRCNN x2 weights and original
known-detail images. No download or TensorFlow installation is required.

## Model and execution

Choose local frozen `.pb` weights, their exact lowercase SHA256, the matching
algorithm and its trained scale. FSRCNN/ESPCN/EDSR offer 2/3/4; LapSRN offers
2/4/8. These selections declare an export contract; they do not identify or
retrain arbitrary weights. Native output dimensions/channels/depth must match.
Only the included FSRCNN x2 CPU export has been qualified in these examples.
Multi-scale inference remains a separate roadmap item, S2-58.

The wrapper checks a bounded byte snapshot and its digest on every inference
activation, even when source file timestamps/length stay unchanged. A cache
miss imports that exact snapshot through an owned temporary file, which is
removed on success or failure. Each block retains at most one private native
model, keyed by digest, algorithm, scale and requested backend/target. Independent
blocks can execute in parallel. Creation is locked around native process-wide
layer registration; ordinary execution stays within the block's executor lock.
Close/reset and native failures release the handle. Caches are never saved.

Unavailable backend/target pairs fail explicitly; the wrapper never substitutes
CPU. Available non-CPU targets still require their own measurements, including
possible native layer fallback. Information reports requested placement, not
observed per-layer execution. CPU runs do not enumerate unused accelerator
devices; a fresh-process background-worker check also verifies clean native
teardown. The native importer uses OpenCV 5's classic DNN
engine. See the [version-pinned native implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/dnn_superres/src/dnn_superres.cpp)
for its model-specific preprocessing/reconstruction; those operations are owned
by the native module rather than editable wrapper normalization settings.

Project-relative models are sealed assets copied byte-for-byte on Save As.
Absolute model paths remain external links. Relative paths cannot escape the
project; copy external weights into the project first for a portable bundle.
The standard JSON/XML persistence retains all declared controls. The model
digest identifies bytes, not their suitability or inspection accuracy.

## Images, selection and coordinates

Accepted encodings are grayscale, BGR/RGB and BGRA/RGBA. Color tags must match
the channel layout. RGB channels are interpreted correctly. Alpha is excluded
from inference and taken from the interpolation baseline; color channels are
treated independently of transparency. Physical depth, labels, HSV and Lab
are rejected: convert ordinary color explicitly before this block.

Default working depth requires uint8. Convert full scale explicitly permits
uint16 or finite float32 [0,1] inputs: full-scale values are rounded/quantized to
8 bits for the native model and restored to the source depth afterward. That
restoration does not recover discarded precision. HDR clipping is not implicit.
The interpolation baseline retains source depth; float baseline overshoot is
clipped to [0,1], with the changed sample count reported. Alpha uses that same
baseline. Original source arrays are never modified.

Mask and Region intersect on the source grid, with known frames checked.
The whole source remains inference context. Selected cells repeat by the
integer scale to choose learned output pixels; outside the selection uses the
baseline, including its original high-depth precision. This is output selection,
not exclusion from model context, a crop or an ROI speed optimization. Empty
selection emits the current baseline with zero processed pixels and no model
read/inference. Connected selection inputs are required when scheduled.

Output-to-input nominal pixel centers follow `(x + 0.5) / scale - 0.5` on each
axis. Geometry updates spatial/pyramid transforms while preserving tickets,
recipes and arbitrary auxiliary payload. Processed mask denotes where learned
pixels were written, not additional measured support. Learned feature alignment
must be qualified separately; the coordinate transform does not establish
physical dimensional accuracy or new optical resolution.

## Results, comparisons and limits

The primary output image is the reconstruction. Baseline can use nearest,
linear, cubic or Lanczos interpolation. Connect either to Compare image quality
with the native-resolution reference for error maps and acceptance limits.
Inference passing means a valid model execution, not fidelity acceptance.
Use a separate inspection/final-judgment block for application criteria.

The retained benchmark reports global error, known dark-feature sizes, lost and
false detail pixels, bead-gap width and flat-region statistics at fixed ROIs and
thresholds. On the clean control, FSRCNN lowers MAE from cubic's 1.7218 to 1.1711
samples, yet loses the one-pixel crack and changes a 73x73 thresholded circle to
71x71. It erases 27 dark-detail pixels versus cubic's 26 and changes 30 samples
in an originally uniform 896-pixel region. On the noisy control, its flat-region
standard deviation is 3.0697 versus linear's 2.3343. A prettier reconstruction
therefore need not be a better inspection image. These findings apply to these
controls, model, sampling and thresholds, not general model ranking.

Resource guards cover model snapshot bytes, source/output pixels and 16384
pixels per output axis before native work. They do not bound native model
workspace or process RSS. Invalid parameters, metadata, unsupported exports or
native exceptions use the common contained error path, with no new output event;
a subsequent valid input/configuration can recover. Timing, logs, ticket outcomes
and input-failure policy remain owned by the shared executor.

Focused checks cover exact native pixels, all five supported encodings,
explicit high-depth quantization, alpha, masks/regions, nominal frame/pyramid
mapping, pinned snapshots, cache reuse/tampering, unavailable targets, native
exceptions, temporary-file cleanup, concurrent independent models and portable
saved projects. The fixtures qualify this software contract, not real-camera
defect sensitivity, accelerator performance or physical equipment acceptance.
