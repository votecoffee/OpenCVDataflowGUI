# Object detection tensor decoding

**Decode object detections** (`ml.decode_detections`) is under **OpenCV → Deep
Neural Networks (dnn) → Object detection**. Connect the original image and the
matching **DNN ONNX inference** `tensors` and `preprocessing` outputs. The decoder
owns no model or native network state; independent instances can run in parallel.
Inference continues to own model caching, explicit engine/target selection and
CPU execution. There are no model downloads during graph loading or execution.

## Explicit output contracts

| Format | Row fields | Typical export |
| --- | --- | --- |
| `xywh_scores` | center X/Y, width/height, one score per class | Decoded YOLOv8-style heads |
| `xywh_objectness` | center X/Y, width/height, objectness, class scores | Decoded YOLOv5-style or raw YOLOX heads |
| `xyxy_score_class` | left/top/right/bottom, score, class ID | Explicit postprocessed detections |
| `ssd` | image ID, class ID, score, left/top/right/bottom | Single-image SSD detection output |
| `ssd_heads` | explicitly named boxes, classes, scores; optional valid count | Separate-head postprocessed SSD exports |

Select rows `[N,C]` or channels-first `[C,N]`, optionally preceded by one batch
dimension of size one. SSD uses `[N,7]`, `[1,N,7]` or `[1,1,N,7]`; negative image
IDs mark unused rows, and positive IDs reject as another batch. Use the
[batch inference adapter](batch-inference.md) and unpack one image before decoding.
Empty detections can use low-score/padding rows or explicitly shaped zero-row
dynamic outputs; inference and decoding preserve those empty tensor records.

**SSD heads** selects three distinct output names. Boxes must be `[N,4]` or
`[1,N,4]`, classes and scores `[N]` or `[1,N]`; sizes must agree and batch size
must be one. Choose **XYXY** or **YXYX** box order explicitly. An optional count
head must be `[1]` or `[1,1]` with an integer in `0..N`. Only the first count
rows are active; padding is ignored after storage shapes, finite values and raw-row
budgets are checked. Without a count head, all rows are active. Packed output name
and layout settings are unused for this format. No head names or coordinate order
are guessed. Empty dynamic heads are supported by the bounded `tensor_v1` path.
The same class-offset/filter, score, source-coordinate, selection, border and NMS
policies apply to packed and separate-head SSD. The dropdown controls are shared
with the inspector and saved projects.

For an actual pretrained export, try the
[pinned SSD example](<../examples/opencv tools/ssd_detection/README.md>).
Its explicit preparation step creates a sealed local model/preprocessing bundle;
Save As copies the bundle and model credits/license. The New-engine CPU path,
native int64 class IDs transported as exact small float32 IDs, independent native
preprocessing/box references, hand-marked cup annotation, failed gates and clean
empty results are checked. The export already performs NMS, so additional
suppression is disabled. This qualifies that export, without claiming arbitrary
SSD anchors, quantized heads, accelerator execution or detector accuracy.
The unsupported Classic import can emit a native Windows fault diagnostic on the
qualified build; its rejection probe runs in a separate process. Sealed settings
reject engine changes before native import. The New-engine worker exits cleanly.

Select network-pixel or normalized coordinates explicitly. **Grid decode:
yolox** requires XYWH/objectness network-pixel rows and an exact declared stride
grid, normally `[8,16,32]`. It adds grid offsets to XY and exponentiates sizes.
Stride divisibility and total row count are checked before grid allocation;
scored log-sizes and resulting coordinates are bounded. Other raw anchor heads,
distributional box regression, segmentation/keypoint heads and
quantized integer tensors need their own adapters. The decoder does not infer
an export schema from a model's name or tensor dimensions.
Oriented boxes have their own [rotated decoder](rotated-detection.md), including
explicit angle conventions, overlap policy and source affine poses.

**Probability** validates class/objectness scores in `[0,1]`. **Sigmoid** explicitly
converts logits; no implicit softmax is applied. Score-head formats select the
highest-scoring class per row. Objectness heads multiply that class probability
by objectness, with a separate objectness gate. Confidence filtering is strictly
greater than the configured threshold, matching native NMS semantics. Labels are
optional; a supplied score-head table must have one entry per class channel.
Explicit-ID formats allow a class offset, e.g. `-1` for one-based SSD IDs. Class
filters apply to the adjusted zero-based IDs; no filter keeps every class.

## Source coordinates, selection and suppression

Preprocessing must identify the current image dimensions and supply inverse,
positive axis-scale/translation matrices. Boxes map through the exact rounded
stretch/letterbox dimensions and padding. Known ticket, spatial-grid and recipe
revision mismatches between the three inputs reject before decoding. Unknown
metadata remains usable for custom adapters; `report.context_checks` identifies
which identity checks were fully available. Matching metadata is not proof of
image content or model revision. Use [model contracts](model-contracts.md) for
versioned model/preprocessing/execution receipts.

Coordinates are continuous **source pixel edges**. A box can reach image width
or height; its size is right minus left, without an extra pixel. Regions/masks
intersect and select the pixel cell containing the box's *unclipped* center.
They filter detections without cropping, moving the source grid or altering the
network's predictions. This is center-based selection, not object-mask coverage.
To exclude image content from inference itself, also wire its mask/region inputs.

NMS runs on floating, unclipped source boxes, after score/class/selection checks.
The default groups classes and calls native `dnn.NMSBoxes` separately, avoiding
cross-class suppression even for negative-coordinate boxes. Class-agnostic mode
uses one group. IoU, native adaptive `eta`, pre-NMS `top_k` and a disable checkbox
are exposed; `top_k` is per class or global in class-agnostic mode. Results sort
by descending score with original row index as the tie breaker. These are model
scores, not calibrated acceptance probabilities.

Border policy clips partly visible boxes, discards them, or preserves their full
coordinates. Fully outside boxes always drop. NMS uses the original boxes before
clipping, so border handling does not change the overlap calculation. Retained
geometry records final XYWH, unclipped/network XYXY, border crossing and actual
clipping separately.

## Collections, graphics, judgments and bounds

The primary image passes through unchanged. `collection` is an ordinary
`vision_collection_v1` box collection, carrying source spatial/pyramid context
and the acquisition ticket/recipe payload. Items have class ID, label, score,
numeric box metrics and corner geometry. Area is rectangle area, not measured
foreground area; orientation is explicitly unmeasured. Existing sort/filter tools
can use width, height, center, area and custom `score`/`class_id` metrics.

Drawable outlines and score/label objects share the source grid. Use
**Overlay result graphics** to render them onto an image, then **Set filmstrip
image** to select that result. Graphics-off/Lean skips object generation while
retaining numerical detections. Minimum/maximum count criteria produce normal
failed inspections with `detection.count`; zero disables the upper bound. Valid
empty collections still emit. Malformed tensors, incompatible geometry and
exceeded budgets emit no current results, through ordinary block containment.
Native exceptions are logged with stack traces and later valid runs can recover.

Image pixels, raw rows, scored/selected NMS candidates and retained results have
explicit limits. Excess candidates/results reject rather than silently truncate
an inspection. The shared tensor reader caps inputs at two million values.
These bounds do not constrain allocations already performed by inference,
runtime snapshots or arbitrary custom plugin code. See all enum choices, numeric
limits and tooltips in the [parameter reference](block-parameters.md).

## Qualified examples and limits

[Nine original control projects and one real-photo example](<../examples/opencv tools/dnn_detection/README.md>)
cover native image-dependent ONNX inference, stretch/letterbox inversion, head
layouts, SSD IDs, class-aware/agnostic NMS, selection and deliberate count failures.
The small MIT control model has fixed proposals and RGB-mean-dependent scores;
it qualifies wiring and geometry, without learned object recognition.

The optional YOLOX-S coffee project uses a size/hash-verified OpenCV Zoo model
at revision `48b10fa505aeb51fbd127fc60a5c660040e823b7`, under its
[Apache 2.0 model license](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/LICENSE).
Its CPU cup result is compared with independently expanded grids and native
batched NMS on the same raw tensor. The unchanged Rachel Michetti coffee photo
retains its [CC0 source record](../examples/assets/photograph-sources.md).
This is inference/postprocessing qualification, not an accuracy benchmark for
industrial parts or a qualification of every YOLO/SSD export or accelerator.

[Twelve YOLO projects](<../examples/opencv tools/yolo_detection/README.md>) now
qualify the pinned YOLOX baseline in sealed model/settings bundles. The original
raw grid export runs under explicit Classic CPU. A reproducible local derivative
appends standard ONNX grid/size decoding, objectness multiplication and transpose,
retaining every learned initializer. Its `[1,8400,84]` and `[1,84,8400]` outputs
exercise decoded rows and channels under explicit New CPU. This derivative is
YOLOX, not a YOLOv8 model or qualification of that model family. Source and derived
bytes have separate pinned digests and a prominent modification record; graph
loading/execution never performs conversion or downloads.

An independent native preprocessing/grid/batched-NMS reference checks source boxes,
class IDs, scores and original row identities. A portrait crop exercises horizontal
padding and unequal rounded scales, complementing the landscape's vertical padding.
Hand-marked cup annotations, class-aware/agnostic suppression, duplicate false
positives, strict confidence/objectness gates, region exclusion, valid empty images,
bad-layout containment/recovery, graphics/Lean, XML, portable credits/model assets
and cold-worker teardown define the qualified scope. The photo and its crop share
one acquisition group and do not establish dataset accuracy. Arbitrary YOLO family,
anchor, quantization and accelerator contracts remain open.

Native references: [OpenCV NMS implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/nms.cpp),
[YOLOX grid/score convention](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/yolox.py).
