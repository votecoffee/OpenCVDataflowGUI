# Semantic segmentation

**OpenCV > Deep Neural Networks (dnn) > Semantic segmentation > Decode semantic
segmentation** (`ml.decode_segmentation`) turns one image's explicitly described
model output into source-aligned labels, a selected-class mask, pixel counts and
per-class statistics. Use ordinary **DNN ONNX inference** to load the model.
See the [twelve runnable projects](<../examples/opencv tools/semantic_segmentation/README.md>).

## Connect and configure

Connect the same acquisition image to inference and the decoder. Connect
inference's **Tensors** and **Preprocessing** to the corresponding decoder inputs.
Configure input encoding, intensity normalization, size and stretch/letterbox
preparation in inference according to the model's actual requirements. The
decoder's optional mask and region select **outputs**, after full-image inference;
selections connected to inference instead zero its unselected input pixels.

Declare a named output and layout; nothing guesses an export's axes or semantics:

| Format | Score meaning | Layouts |
| --- | --- | --- |
| `multiclass` | Mutually exclusive class logits (`softmax`) or probabilities summing to one within 1e-4 | NCHW, NHWC, CHW, HWC; HW/NHW also describe one channel |
| `binary` | One foreground probability (`probability`) or foreground logit (`sigmoid`); class 0 is background, class 1 foreground | Any listed layout, with exactly one channel |
| `class_ids` | Already decoded integer IDs, or -1 for unknown | Any listed layout, with exactly one channel |

Batch N must be one. For batched inference, use **Select inference result** or
**Unpack inference results** to deliver the corresponding image, tensors and
preprocessing together. Heads must use dequantized float32 `tensor_v1` records.
Decoded IDs require **Class count** (1..256); score channels infer it when zero,
while binary always has two classes. An explicit score class count must match.
The labels table, when supplied, has one name per class including background.

**Selected class IDs** controls the mask and area judgment. The default `[1]`
selects class 1; `[]` selects all nonignored classes, including class zero unless
explicitly ignored. **Ignored class IDs** converts those winning IDs to -1;
it never assigns their pixels to a runner-up class. Class zero remains a valid
class in the label plane. IDs denote semantic categories, not connected objects.

**Minimum confidence** accepts winning confidence >= the threshold. Multiclass
confidence is the winning softmax/probability. Binary uses foreground probability
when class 1 wins the configured **Binary threshold**, and `1-p` for class 0.
Changing the binary threshold can therefore select a foreground class whose
confidence is below 0.5. Softmax is numerically stable for extreme finite logits.
Probabilities are checked throughout the raw head, even outside sampled content.
Unknown decoded IDs have no confidence: the confidence output is absent, the
result score is absent, and minimum confidence must be zero. A supplied ID is
never treated as a probability-one observation.

## Source grid and sampling

The output head is assumed to uniformly cover the **entire network input field**.
This explicit adapter does not represent cropped/offset/receptive-field-specific
heads or multilabel independent sigmoid channels. Such exports need a matching
adapter; a matching tensor shape alone does not establish correct image geometry.

The decoder validates preprocessing dimensions, inverse positive axis-scale/offset
maps, and source coverage within the network field. It uses the exact rounded
letterbox scales/padding from inference, not an idealized common scale. For source
pixel `(x,y)`, map `(x+0.5,y+0.5)` through **Input to network**, then multiply by
`head_size/network_size`. These are head **edge** coordinates.

- **labels_nearest**, the default, classifies on the head grid and takes
  `floor(edge_coordinate)` per source pixel, clamping the outer head boundary.
  Exact half-cell ties choose the cell to the right/below. Confidence follows
  the same nearest cell. Label IDs are never blended.
- **scores_linear** subtracts 0.5 to express head pixel-center coordinates and
  samples each raw score channel with OpenCV `remap`, `INTER_LINEAR` and replicated
  head-edge borders, then applies activation/argmax/threshold on the source grid.
  This can change a boundary relative to nearest winner labels. Native floating
  interpolation precision depends on the installed implementation; the included
  scalar bilinear references qualify the supported CPU wheel. Decoded IDs require
  nearest resampling. Padding predictions can contribute near a content boundary
  under linear interpolation; they are not independently measured source pixels.

Known ticket, spatial-frame and recipe conflicts among image/tensors/preprocessing
reject the invocation. Unknown identities remain unknown; the report identifies
whether all three tickets/frames were available. Image encoding/depth, ticket,
recipe, pyramid level, coordinate transforms and processing start are retained
from the source image on every emitted output. No scaling to an unrelated largest
pyramid level happens silently.

## Outputs, judgments and graphics

| Output | Contract |
| --- | --- |
| Image | Original pixels, encoding and depth |
| Labels | Source-size signed int32 class IDs; -1 means ignored, unknown, outside the ROI/mask, or below confidence |
| Mask | Source-size uint8, 255 for a selected valid class, otherwise zero |
| Confidence | Source-size float32 winning confidence, zero at excluded pixels; absent for decoded IDs |
| Pixel count / Fraction | Selected pixels and selected/full-ROI area fraction |
| Report | Class pixel counts, fractions, mean winning confidence, selection flags, mapping/format, context checks and graphics-limit status |

The ROI denominator is the full intersection of optional region and mask, before
ignored-class or confidence rejection. Those rejected pixels remain in the
denominator. An empty ROI has fraction zero. Count/fraction limits create a normal
`segmentation.area` fail judgment while retaining the current outputs; minimum
pixels zero permits a valid empty result. Result score is the mean confidence of
selected pixels, or absent when no such confidence exists. These are model
confidence/area decisions, not an application-specific defect verdict.

Feed **Mask** to Find blobs, contours, morphology or region measurement for further
inspection. Semantic IDs can span multiple disconnected objects; use connected
components when each object needs its own identity. Legacy label-measurement tools
may reserve zero for background, so do not assume they interpret class zero as a
measurable semantic class.

Optional diagnostics include a semantic color map (ignored pixels black) and
drawable selection outlines, holes and an area summary. Drawings use pass/fail
colors and stay separate from the numerical images. Connect hidden **Result
graphics** to **Overlay result graphics**, then **Set filmstrip image**, to use the
annotated source image in the filmstrip. Graphics contour/point limits report
truncation without changing numerical masks. Graphics Off/Lean skips the map and
outlines entirely.

The tool is stateless and supports isolated previews and parallel branches. The
base executor owns timing, history, events, input failure policy and exceptions;
an exception publishes no current result, and valid subsequent input can recover.

## Bounds and qualification

Heads are limited to 256 channels, 4096 pixels per axis and a configurable maximum
of eight million serialized values. Source images are limited to 16384 per axis
and the configured pixel budget. Linear resampling also checks channels x source
pixels before allocating mapped scores. Inference has its own total-output limit;
increase it deliberately when a valid segmentation head exceeds the default.
These checks bound retained decoding work, not a model's internal allocations.

Original image-dependent color-arithmetic ONNX fixtures qualify decoding, source
mapping, confidence/class semantics, empty/failing inspections, blob handoff,
graphics, JSON/XML and Save As. Their independently calculated PNG label references
are numerical test truth, not trained industrial segmentation accuracy. No models
are downloaded during a graph run. Promptable models, arbitrary export support, accelerators and task-specific
recognition accuracy remain separate qualification. [Instance segmentation](instance-segmentation.md)
now covers declared ROI and prototype heads.

Upstream references: [OpenCV semantic-segmentation sample](https://github.com/opencv/opencv/blob/5.x/samples/dnn/segmentation.py)
for per-pixel class decoding, and [OpenCV image-transform declarations](https://github.com/opencv/opencv/blob/5.x/modules/imgproc/include/opencv2/imgproc.hpp)
for remapping. This adapter's frame, selection and judgment contracts are defined
above.
