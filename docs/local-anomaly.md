# Local image anomaly references

Use **OpenCV → Deep Neural Networks → Local anomalies** to build a known-good
reference bank from local ONNX features and inspect new images. Start with the
[twelve examples](<../examples/opencv tools/local_anomaly/README.md>).

## Features and training

`ml.local_features` accepts the original Image, Tensors, Preprocessing and Receipt
from existing ONNX inference. Select an explicit NCHW or NHWC spatial output with
N=1, 2..4096 cells and 1..256 channels. Features are float32, bounded to +/-1e6.
Choose unchanged feature units or per-cell L2 normalization; zero L2 vectors are
invalid. A pooled whole-image descriptor is not a spatial feature map.

The grid convention is **uniform cells over the entire network field**. The
preprocessing maps restore pixel-edge coordinates to the source; the feature
export must actually obey this convention. Receptive-field offsets, feature
stride/crop conventions and model semantics are not inferred from shape alone.
An arbitrary intermediate layer needs its own export/geometry qualification.

Optional rectangle/circle/polygon regions and masks select cell centers. Padding
centers outside the original image are excluded. Model receptive fields can still
see excluded pixels. This is selection for reference membership, not masked
inference or proof that a feature describes only its center pixel.

Known tickets, frames and recipes must agree. Receipts validate model/output and
preprocessing identity; detailed single-image output hashes are checked. Unpacked
batch-row receipts cannot verify the original full-batch output hash. The source
image's pixel/depth/shape hash is recorded and later checked against the inspection
image. This associates the supplied image and grid but does not independently
prove that inference consumed those pixels or that camera exposures were synchronized.

`ml.train_local_anomaly` accepts one grid, a list or a FlowGroup of up to 64
known-good grids. Model revision, output head, dimensions, layout, normalization,
grid and preprocessing contracts must agree. Duplicate source pixel hashes are
rejected. The caller labels the images as known-good; the tool cannot verify those
labels or independent acquisitions.

Choose Euclidean or Manhattan distance. All selected rows are retained by default;
overflow rejects the training rather than silently discarding patches. Explicit
Even stride sampling retains raster/source-ordered rows at evenly spaced indices.
This is deterministic reduction, not a representative coreset or PatchCore.
Changing sampling changes the content revision and requires threshold review.

## Inspect and interpret

`ml.local_anomaly` compares each valid query cell with every retained reference
using native `cv2.batchDistance`, in bounded query batches. Search is global across
the reference bank, so it can miss rearrangements of otherwise familiar textures.
It does not enforce part alignment, object identity or positional consistency.

Nearest-cell resampling maps raw distances onto the original image grid. Image
pixels/depth/encoding remain unchanged. Distance is float32; Support and Mask are
uint8 source-grid images. Zero outside Support is unscored and must not be read as
evidence of a normal pixel. Inspection masks/regions restrict scored output; no
support or insufficient selected support is an error.

Pixels with distance **greater than** Distance threshold are anomalous; equality
is accepted. The inspection passes when their source-grid count is at most
Allowed anomaly pixels. These are ordinary inspection failures with reason
`anomaly.area`; no final ticket judgment or normalized confidence is invented.
Maximum distance, all native cell distances, nearest reference indices, scored
support, reference revision and decision revision remain in the report. A tie's
reference index follows native ordering; the distance remains the relevant result.

Drawable contours/labels can be aggregated over the unchanged source image. The
optional heatmap uses a fixed threshold-relative scale, rather than normalizing
each image independently. Graphics limits can truncate/simplify diagnostic outlines;
the full numeric mask and judgments are unchanged. Disabling graphics preserves
the numerical outputs and decisions.

Reference size, dimensions, candidate rows, image pixels and distance work have
explicit bounds. Work is query cells × reference rows × dimensions. A budget error
emits no current result rather than an incomplete nearest-reference search.

## Persistence and limits

`ml.save_local_anomaly` creates a JSON bank without overwriting an existing file.
`ml.load_local_anomaly` validates strict JSON, numeric/shape/normalization bounds
and a SHA-256 content revision. A loaded reference is a sealed Save As asset;
JSON/XML projects retain its identity and the ONNX model bundle. These are data
files; loading does not select executable code.

The examples train on three original synthetic surfaces and query different noise
seeds. An original ONNX convolution encodes each 2×2 cell as its four pixel values.
It is a transparent CPU integration fixture, not a learned industrial backbone.
An 8×8 dark defect and removed texture both localize to exactly 64 pixels. A weak
one-pixel defect is missed at threshold 0.2; a global brightness change rejects all
1024 pixels despite having no labelled structural defect. These controls show
feature/threshold dependence rather than production accuracy.

Independent NumPy patch extraction and exhaustive distances verify the native
outputs. Training/query byte separation, exact regeneration, Debug/Lean, high depth,
valid failures, invalid identities/models, reference I/O, JSON/XML/Save As,
failed-input gates and cold Run All shutdown are covered. Pretrained anomaly
backbones, dataset-level false-reject/recall estimates, accelerator behavior and
physical inspection acceptance require separate evidence.
