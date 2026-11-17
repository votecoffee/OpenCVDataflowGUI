# Instance segmentation

**OpenCV > Deep Neural Networks (dnn) > Instance segmentation** contains three
stateless blocks: **Decode instance segmentation** (`ml.decode_instances`),
**Pack separate detection heads** (`ml.pack_detection_heads`) and **Select instance
mask** (`ml.select_instance_mask`). The [twelve projects](<../examples/opencv tools/instance_segmentation/README.md>)
run with the base installation and a tiny original ONNX control model.

## Detection and mask contracts

Connect the same original image to **DNN ONNX inference** and the decoder, then
connect inference's **Tensors** and **Preprocessing** to the decoder. Explicitly
select the detector output format, axes, coordinate units, class names and
score/objectness thresholds. Ordinary detector settings also control class
filtering, class offsets, class-aware/agnostic box NMS, eta/top-K, border policy
and work limits. See [detection contracts](object-detection.md).

Mask rows always refer to **original detector rows before filtering or NMS**.
Suppressed rows are not renumbered. This avoids assigning a different object's mask
after a score sort or suppression. One image is decoded per invocation; use the
existing batch result selector/unpacker to associate image, tensors and preparation.

| Mask mode | Mask tensor | Detection tensor |
| --- | --- | --- |
| `roi` | NCHW/NHWC with C=1, or NHW; N is raw detection count | Ordinary explicit XYWH, XYWH/objectness, XYXY/score/class or SSD rows |
| `roi_per_class` | NCHW/NHWC; channel = final detection class ID + Mask class offset | Same row formats; offsets handle exports with a background channel |
| `prototypes` | CHW/HWC or batch-one NCHW/NHWC; C is prototype count K | XYWH/class scores or XYWH/objectness/class scores followed by exactly K coefficients |

ROI heads may retain one additional leading image-batch axis of size one, for
example `[1,N,C,H,W]` after batch unpacking. A non-one image batch rejects.
Prototype mode infers K from its explicitly configured tensor layout and removes
the trailing K coefficient columns before ordinary detection decoding. Coefficients
and prototype channels are never interpreted as class probabilities. A class
offset applies before selecting the ROI class channel; configure the mask channel
offset separately. Detection and mask tensor names must differ.

**Mask activation** independently declares probabilities or sigmoid logits.
**Activation order** chooses activation on the raw mask grid before resampling,
or activation of sampled logits afterward. These operations can give different
boundary pixels. **Mask threshold** uses `probability >= threshold`.
**Mask interpolation** chooses nearest cells or OpenCV linear sampling. Working
mask/mixed-score planes use float64; the supported CPU linear path quantizes map
phases to 1/32 pixel. This differs from some float32 remapping paths. The examples
and scalar references qualify the installed implementation rather than assume
identical interpolation across builds.

In ROI modes, each mask uniformly covers the **original unclipped box**, with
half-open continuous pixel edges. Only source pixel centers inside that box are
sampled. Clipping at an image edge removes outside pixels; it never stretches
the remaining mask to fit a smaller box. Mask coordinates are therefore preserved
through the exact inverse resize/rounded letterbox mapping.

Prototype planes uniformly cover the entire network input field. The decoder
linearly combines a retained row's coefficients with those planes, then samples
the matching source pixel centers through the recorded maps. **Crop to box**
limits prototype masks to the original predicted boxes by default; disabling it
keeps their full source-frame support. Cropped/offset prototype fields, rotated
instance boxes, export-specific cropping/threshold orders and multilabel heads
need a matching adapter. These controls do not establish that an arbitrary export
uses this precise mask convention.

## Selection, counts and outputs

Optional mask and circle/rectangle/polygon/raster region inputs operate in the
source image frame. A disconnected selection uses the whole source; a connected
selection with no current value prevents execution rather than falling back to
unmasked processing. **Selection mode** chooses:

- **pixels**: intersect decoded mask pixels after box filtering/NMS; a box center
  may lie outside the selected area.
- **centers**: filter original box centers before NMS, retaining full decoded masks.
- **centers_and_pixels**: apply both behaviors.

**Minimum/Maximum mask pixels** drops instances outside the selected source-pixel
area range; maximum zero disables the upper filter. Minimum zero preserves valid
empty masks. **Minimum/Maximum count** evaluates the final retained instances
after this area filter. An out-of-range count is a normal `instance.count`
inspection failure with complete current outputs, rather than an exception.

| Output | Meaning |
| --- | --- |
| Image | Original source pixels, encoding and depth |
| Collection | Ordinary sortable/filterable box collection, with an exact `mask_region` and `label_id` on every retained item |
| Count | Instances after box selection/NMS and mask-area filtering |
| Mask | Source-size uint8 union, 255 wherever any retained mask is foreground |
| Labels | Source-size int32 summary: zero background, otherwise original detection-row ID + 1 |
| Overlap count | Source-size int32 number of retained masks covering each pixel |
| Report | Detector settings/results, mask contract, removed mask areas, work counts, label-ID mapping and diagnostic truncation |

Each `mask_region` is an `image_region_v1` raster with sorted source-grid scanline
runs. It remains separate from every other mask, including at **overlapping
pixels**. The top-level `instance_masks` marker declares the instance collection
contract. Sort/filter blocks preserve this marker and item masks. **Label overlap**
resolves only the convenient summary label image by detection score, mask
confidence, or their product; ties retain the earlier stable detector order.
It never removes pixels from an item's own region. Summary labels cannot be used
to reconstruct every original mask where objects overlap.

Existing standard `area`, position and dimensions remain box measurements.
Additional numeric metrics are `mask_area`, `mask_mean_confidence`,
`mask_center_x`, `mask_center_y` and `mask_coverage` (selected pixels divided by
the sampled source tile area). Choose **custom** and the desired key in collection
sort/filter blocks. Mask mean confidence is distinct from detector confidence.
Areas/centroids use the current image/pyramid level; known transforms remain in
payloads. Predicted mask overlap is not an occlusion measurement or proof that a
part is physically pickable.

Drawable outlines, holes and ID/class/score/area labels use stable per-instance
colors. They remain objects separate from numerical images. Connect hidden
**Result graphics** to **Overlay result graphics**, then **Set filmstrip image**
to annotate filmstrip results. A label-map diagnostic is also available.
Graphics limits report omissions while leaving complete numerical regions intact.
Graphics Off/Lean skips optional drawings and diagnostics.

## Separate model outputs and individual inspection

**Pack separate detection heads** accepts one named tensor dictionary containing
XYXY boxes `[N,4]` or `[1,N,4]`, scores `[N]`, and integer class IDs `[N]`.
Score/class vectors may have singleton axes. Configure their actual names. The
block adds a new `[1,N,6]` XYXY/score/class tensor and preserves original heads,
including masks. Connect its Tensors to the decoder, configure format
`xyxy_score_class` and select the new output name. This adapter also works with
ordinary **Decode object detections**. Coordinate units and ID offsets are handled
by the decoder, not guessed by the packer. Existing output names, invalid values,
inconsistent row counts and work-budget excess reject atomically.

**Select instance mask** accepts the original image and an instance collection,
optionally after sorting/filtering. Select a zero-based collection index or the
original detector row ID. It emits the exact uint8 mask, source-grid region,
original item, original image and Found boolean. Connect its mask to blob/contour
tools, or its region to an ordinary regional inspection block. Missing selection
either produces an empty failed result with `instance.missing` (Item absent), or
an exception with no current outputs, according to **Missing**. No fallback to an
old selected mask occurs. Pixel/run limits bound rasterization.

Known source image/tensor/preprocessing ticket, recipe and frame conflicts reject.
The selector checks the collection's size, origin, spatial context, ticket and
recipe against its image. Unknown identities remain unknown. Image/depth/channel
encoding, ticket, pyramid/grid transforms, recipe and processing start are retained.

## Empty tensors, bounds and qualification

Inference output serialization preserves correctly shaped zero-element tensors.
Detector/instance heads accept zero **rows**, with valid remaining axes; ROI mask
row counts must still match. Separate-head packing and batch result unpacking
preserve this empty shape and source ownership. Empty detections produce current
zero masks and a normal count verdict. Generic input tensor creation,
classification/embedding and other consumers retain their previous nonempty rules
unless they explicitly opt into empty output reading. Native engines can still
reject unsupported zero-shape operators before a result exists; serialization
support does not make those models executable.

Mask/prototype heads have configurable serialized-value limits, up to eight
million values, with up to 256 channels and 4096 pixels per spatial axis. Source
pixel/axis, raw-row, candidate and result limits remain explicit. Source tiles
are bounded before resampling; prototype multiply work is bounded before mixing.
The decoder uses one source tile at a time instead of allocating N complete mask
images. Total scanline runs are bounded across all retained instances; excess
rejects without publishing partial masks. These limits bound decoder retention,
not internal native network allocations.

The base executor owns one invocation's history, timing, events and diagnostics.
Native failures publish no current result, and valid subsequent input recovers.
The blocks are stateless, preview-safe, parallel-safe and input-readonly.

Native image-dependent ONNX projects and independent packed-pixel references
qualify original-row association, ROI clipping, class masks, coefficient mixing,
layout/preparation, overlap, selection, counts, graphics, JSON/XML and Save As.
Zero-row data and stubbed native-array handoffs separately qualify serialization,
receipts and batch ownership. No native model is claimed to produce zero rows
based on those stubs. The original fixture has fixed boxes/templates and image
dependent scores/gain; it does not establish trained recognition accuracy,
arbitrary export compatibility, accelerator support or physical pose accuracy.

Primary references: [OpenCV Mask R-CNN sample](https://github.com/opencv/opencv/blob/5.x/samples/dnn/mask_rcnn.py)
for a model returning separate boxes/classes/scores/ROI masks, and
[ONNX operators](https://onnx.ai/onnx/operators/) for tensor arithmetic. The
examples' numerical model, assets and reference calculations are original work.
