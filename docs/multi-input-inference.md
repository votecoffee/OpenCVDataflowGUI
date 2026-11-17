# Named multi-input ONNX inference

Use three tools under **OpenCV → Deep Neural Networks (dnn)**:

| Tool | Purpose |
| --- | --- |
| `ml.image_tensor` — Image to DNN tensor | Independently prepare each image's size, color, normalization, selection and layout. |
| `ml.make_tensor` — Create float32 tensor | Shape a scalar/flat numeric or boolean input, or configured constant, into a portable tensor. |
| `ml.dnn_multi` — Multi-input ONNX inference | Bind up to four tensors to exact model input names and run one complete activation. |

The existing `ml.dnn` single-image interface remains available. Both inference
blocks use the same native model loading, engine/target/fusion settings, output
selection, bounds and handle lifecycle. No additional runtime package, model
download, camera or device is needed for the [nine examples](<../examples/opencv tools/multi_input_inference/README.md>).

## Image and numeric preparation

**Image to DNN tensor** shares the single-input preprocessing implementation:
`(pixel - mean) * scale / std` in the selected RGB/BGR/gray order, with stretch
or centered rounded letterbox resize. Source intensity units are preserved before
normalization; a uint16 image does not silently become an 8-bit image. Mean/std
must match channel count, remain representable in float32 and use positive std.
NCHW `[1,C,H,W]` and NHWC `[1,H,W,C]` are explicit layout choices.

Masks/regions intersect on the source grid. Excluded samples become zero before
resize/normalization; source dimensions and pixels remain unchanged. The image
output is that original image, while the `tensor` output is a float32 `tensor_v1`
record. `preprocessing` records the source/network maps, layout, dtype and tensor
shape. The tensor payload's `tensor_preprocessing` carries the same information
along with the original ticket, recipe and spatial/pyramid context.

**Create float32 tensor** takes a scalar or flat list from Data, falling back to
Values when Data is absent. Shape must have one to eight positive axes whose
product equals the value count. Booleans explicitly widen to zero/one; numeric
values round to float32 and overflow/nonfinite samples reject. Configured constants
have no invented acquisition ticket. Use an explicit Reference role when a model
consumes such a constant alongside a live image. The current tools do not emit
ONNX int64/bool/string input tensors or zero-dimensional scalars.

Source pixels and prepared/combined tensor element counts have declared budgets.
Prepared size is checked before resizing, and tensor records are checked before
float32 conversion. These guards do not bound runtime snapshots, model weights
or the native network's internal allocations.

## Complete named bindings

Choose Active inputs, then configure the first N slots. Each has an exact model
input name, expected shape, optional Tensor name and Current/Reference dropdown.
Input names must be distinct. A single tensor record or one-entry map needs no
selector; a multi-output map requires the exact desired output name. This prevents
silently feeding the first arbitrary model output into another network.

An expected axis of `-1` permits variation only if the native model supports it.
The declaration is a caller contract, not automatic ONNX signature discovery or
permission to reinterpret layout. Every active slot must arrive for every
activation. Missing, inactive-extra, malformed, wrong-dtype or mismatched tensors
emit no current outputs. The block does not cache missing input values or infer
from a partial set. Model inputs are bound by name with `Net.setInput`, without
renaming the network's declared inputs.

Valid unchanged signatures reuse a native handle. Changed model/execution
settings or actual named input shapes rebuild it. Binding/forward failures discard
the handle; a later complete activation can recover. Every successful activation
rebinds every configured input. No native network is serialized, copied between
workers or shared by independent block instances. The block is not declared
parallel-safe; independent instances can use the normal scheduler.

This wheel's OpenCV 5 forward method requires the complete terminal-output set.
Both inference blocks fetch it once, then retain the requested terminal subset in
the requested order. Unknown output names reject. The retained-output budget does
not bound unselected native output allocations, and intermediate layer selection
is outside this wrapper contract.

## Tickets, references and output ownership

| Ticket policy | Current input rule |
| --- | --- |
| `same_ticket` (default) | Match the application's normalized ticket key, including its session. External/fixed tickets already share the external session by design. |
| `same_value` | Explicitly pair kind/value across sessions; equivalent timestamp timezones normalize. |
| `unchecked` | Deliberately disable ticket matching; the receipt says verification was disabled. |

If any Current input has a ticket, checked modes require every Current input to
have one. Completely unticketed custom tensor workflows remain possible, with
`tickets_verified: false`. Reference inputs explicitly bypass *only* live-ticket
matching, retaining their own recorded identities. They do not bypass ordinary
input failure policies or the shared runtime's recipe-revision consistency rules.
Different camera grids are allowed: matching tickets do not imply the same view.

Primary input must be an active Current slot. It owns output ticket and spatial/pyramid metadata; recipe revisions follow
the shared runtime. Its image-preparation maps become the optional
`preprocessing` output; a non-image primary without those maps emits no geometry.
Input preprocessing is not mislabeled as output-tensor layout: the known
`tensor_preprocessing` key is removed from result tensor payloads. `input_info`
retains per-slot names, roles, actual shapes/dtypes, tickets, recipes, grids and
preprocessing, with the selected policy and total input element count. Prepared
tensor shape/type and known source dimensions must agree with their provenance.

This block does not wait for cameras, pair exposures, buffer a history or certify
physical synchronization. Use [input coordination](input-coordination.md),
queues or an appropriate shared trigger upstream. A repeated filename or PLC
value is only as unique as the source's ticket convention. Value matching does not
merge the ticket ledger; use a shared ticket identity upstream when one ledger
entry must cover both acquisitions. Batching and dynamic
image-set inference remain separate S2-04 work; revisioned model/preprocessing
receipts remain S2-05.

## Qualification

Original MIT controls use native ONNX RGB/gray subtraction, image/gain
multiplication and four-branch sums, with independent arithmetic references.
NCHW/NHWC, differently sized channel axes, explicit named order, two source grids,
primary ownership, cached references and deliberate shape/ticket failures are
checked. These are numerical image-application fixtures without learned
recognition or industrial accuracy claims. Native reference:
[OpenCV Net input/forward contract](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/include/opencv2/dnn/dnn.hpp).
