# Batched and dynamic-shape image inference

**Batched ONNX image inference** (`ml.dnn_batch`) is under **OpenCV → Deep Neural
Networks (dnn)**. Connect an ordered image `FlowGroup`, for example from **Batch
serial entries** or a TIFF page decoder. The block processes that complete group
in bounded native calls. It does not acquire images, wait for a future member,
flush an unfinished upstream collection or change the graph's scheduling policy.
Use [control flow](control-flow.md) for collection and buffering.

The existing single-image and [named multi-input tools](multi-input-inference.md)
remain available. This adapter binds one float32 NCHW image input. General typed
or paired multi-input image batching is outside its contract.

## Shapes and tails

Declare the exact model input name and `[N,C,H,W]` signature. Positive axes must
match every submitted call; `-1` permits variation only where the model/backend
actually accepts it. The declaration is not automatic ONNX signature discovery.
Some simple native operators accept dimensions different from an export's stated
shape; caller-declared fixed axes are therefore checked before preprocessing.

Configured spatial mode resizes each source to Width/Height. Mixed original sizes,
depths and channel encodings can share a batch when preprocessing converts them to
the same network shape. Source mode keeps the original dimensions, requires one
common size per group and permits a different common size on a later activation.
Network axes are limited to 4096. No implicit crop or reinterpretation occurs.

Batch size bounds each native call, preserving stored group order across chunks.
For five images and a batch size of two:

| Tail policy | Submitted calls | Returned image rows |
| --- | --- | --- |
| Variable | 2, 2, 1 | 5 |
| Pad repeat | 2, 2, 2; duplicate the final real image | 5; discard the padding output |
| Reject | Invalid activation, no native calls | None |

Padding is explicit because batch-coupled models can change their real predictions
when a repeated image is added. It is not guaranteed equivalent to a smaller
batch. No padded acquisition, result or ticket is invented.

## Preprocessing and selections

Every real image uses the existing DNN preprocessing: explicit RGB/BGR/grayscale,
source intensity units, `(pixel - mean) * scale / std`, stretch or centered rounded
letterbox, and source/network coordinate maps. A uint16 image does not silently
become uint8. Alpha/color conversion follows the existing image conversion rules.
Native `blobFromImages` assembles those already normalized images without further
scaling or channel swapping. The adapter does not introduce new interpolation or
normalization defaults.

Optional Masks and Regions are same-length, same-label, same-order groups. Known
per-member tickets and grids must agree with the corresponding image. Missing
identity is not proof of alignment; duplicate labels alone cannot certify pairing.
Masks and regions intersect on each source grid. Excluded samples become zero
*before* normalization, while original pixels and dimensions remain unchanged.
This suppresses input evidence rather than changing result coordinates.

Source image count, per-image/combined source pixels and prepared tensor elements
have explicit limits. Shape/tail/tensor budgets are checked before conversion or
batch allocation. The retained-output limit applies cumulatively across native
calls, including padded rows. These limits exclude graph snapshots, weights and
native internal or unselected-output allocations. All terminal outputs are fetched
once per native call before requested name/subset ordering, as in the other DNN
blocks. The adapter splits native arrays directly, avoiding a full-batch list
serialization followed by reconstruction.

## Results, tickets and failures

Results is an ordered `FlowGroup` with the original labels. Each entry carries an
`image_inference_result_v1` record containing the original image, zero-based group
index, label, single-image named `tensor_v1` outputs and that image's preprocessing
map. The image is a runtime array, so the combined record uses an Any port rather
than pretending to be plain JSON. Receipts describe its image shape/type. The
entry payload preserves its source ticket, spatial/pyramid information, companion
data, recipe and processing start time. Aggregate Info has no invented common
image grid. It lists submitted shapes, chunk start indices, real/padded counts and
retained elements.

Use **Select group entry** to inspect one row, or **Unbatch serial entries** to
emit all rows. Then connect **Unpack image inference result** (`ml.unpack_inference`)
to expose Image, Tensors, Preprocessing, Index and Label. Existing classification,
embedding and detection decoders can consume the resulting single-image outputs.
Each native output must retain a leading batch axis equal to the submitted count.
The wrapper cannot certify a model's semantic ordering or per-row independence.

Default failed-input behavior can continue processing a group, retaining each
member's chain status and failure reasons independently. A failed sibling does
not mark other rows failed. Setting Images to Reject failed rejects the whole
activation without changing the last processed input/result or native state.
Inference does not assign a final inspection judgment. The shared recipe system
and an explicit group check reject conflicting recipe revisions.

Malformed inputs, unsupported native shapes, lost batch axes, unavailable targets
or a failure in any later chunk produce no current result events. Successful
earlier chunks in that activation are not published. Binding/forward/process
failures discard the native handle; complete subsequent inputs can recover.
Actual input signatures participate in cache identity and every call rebinds its
input. One handle is retained per block instance, released on reset/close and
omitted from JSON/XML. Native process crashes are outside Python exception
containment. Accelerator support and arbitrary model operator coverage require
their own qualification.

## Examples and qualification

[Ten original MIT projects](<../examples/opencv tools/batch_inference/README.md>)
include native dynamic RGB/grayscale pooling, fixed two-row reshaping, mixed-size
inputs, source-size inference, serial downstream execution and four deliberate
invalid activations. Independent channel means verify order and values; tolerance
is 2e-6 for 8×8 controls and 2e-5 for native 40×24 float32 accumulation. These
arithmetic models have no learned recognition or throughput claims. Native API:
[OpenCV 5 DNN input and batch contracts](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/include/opencv2/dnn/dnn.hpp).
