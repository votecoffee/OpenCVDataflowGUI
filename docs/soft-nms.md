# Score-decay detection suppression

**Suppress overlapping detections (Soft-NMS)** (`ml.soft_nms`) is under
**OpenCV > Deep Neural Networks (dnn) > Object detection**. Connect the source
image and a zero-origin `vision_collection_v1` box collection. For raw model
proposals, turn **NMS enabled** off in **Decode object detections** first.
Suppression cannot restore proposals removed by an earlier block or model.

## Methods and native parameters

| Method | Behavior |
| --- | --- |
| Linear | Native `dnn.softNMSBoxes`; multiply a remaining score by `1-IoU` when overlap is above IoU threshold. |
| Gaussian | Native `dnn.softNMSBoxes`; multiply by `exp(-IoU²/sigma)`. IoU threshold is ignored by this native method. |
| Hard | Native `dnn.NMSBoxes` comparison using the same integer boxes; remove strongly overlapping candidates. |
| None | Keep candidates by score/class/selection and sort them; skip native suppression calls. |

Every iteration selects the current best score, so decay can reorder proposals.
Scores, threshold, overlap and sigma follow native float32 arithmetic. Soft/None
retain scores equal to the threshold; native Hard requires strictly greater.
With threshold zero, Soft can retain a fully decayed zero score. Choose a positive
threshold to reject these. Sigma must be positive (1e-6..100); its default is 0.5.
Scores must be probabilities in [0,1], not arbitrary logits or calibrated
acceptance probabilities. Use the tensor decoder's explicit score activation first.

Classes are suppressed independently by default, with an optional accepted-class
list. **Class agnostic** uses one group. **Top k** limits retained results per
group, zero meaning unlimited. The Hard comparison performs full native
suppression before applying this retained cap; it does not use native Hard's
different candidate-prefilter `top_k` interpretation. Final results sort by updated
score, breaking ties by original collection index. Native equal-score picks use
the same original order within each group, including negative-coordinate boxes.

## Geometry and selection

OpenCV 5's Python Soft-NMS binding accepts integer `Rect` boxes. **Box source**
defaults to required `geometry.unclipped_xyxy`; explicitly select **bbox** to use
the visible/output XYWH rectangle instead. There is no silent fallback. This
choice affects overlap around image borders. No clipping or coordinate transform
is performed here; use existing mapping/fusion tools before suppression.

**Rounding** converts only the native suppression rectangle:

- **Cover** floors left/top and ceils right/bottom.
- **Nearest** rounds each edge to the nearest integer, ties to even. If that
  collapses a box, execution rejects instead of silently dropping it.

Both methods leave all output geometry/metrics other than score unchanged.
The report and retained-item `soft_nms` field record the actual integer XYWH used.
Fractional geometry therefore does not imply fractional-IoU suppression. Native
dimensions are limited to 32,767 pixels each, with coordinates bounded to ±1e7;
this keeps integer rectangle areas and pairwise area sums representable.

Optional masks and circle/rectangle/polygon regions intersect and select the
pixel cell containing the chosen source box's center before suppression. This
filters candidates; it does not crop inference or measure object-mask coverage.
Without a selection, existing off-image proposals remain eligible. Primary
image pixels, depth and encoding pass through unchanged.

## Results, provenance and failures

**Collection**, **Indices** and **Scores** are aligned in final ranked order.
Indices refer to positions in this block's input collection, which may already
be sorted by a decoder. Item IDs and class IDs are preserved. `score` and
`metrics.score` become the decayed score; the item's `soft_nms` record stores
its original score, input float32 score, source index/ID, class and native box.
Repeating suppression replaces that block-owned observation field; other item,
geometry, metric and collection fields are retained. The optional candidate trace
also shows score/class/selection removals. Native Soft-NMS does not return final
scores for removed candidates, so they are null and labelled
`suppressed_or_top_k`, without inventing a more specific reason or zero score.

Source image, collection context and wire metadata must agree wherever known:
image size/origin, spatial root/grid/root size, pyramid, acquisition ticket and
recipe revision. Unknown metadata remains usable for custom adapters; the report
identifies available identity checks. Known context follows outputs. Metadata
agreement does not establish source content or physical acquisition synchrony.

Drawable retained outlines/updated-score labels are green; removed selected
candidates are red. Use **Overlay result graphics** and **Set filmstrip image**
to render them in the filmstrip. Lean/graphics controls skip optional drawings
when they are not needed. **Include candidate details** independently controls
the bounded trace; indices/scores and retained-item provenance remain available.

Minimum/maximum count criteria produce ordinary inspection failures with
`suppression.count`. Valid empty collections emit. Invalid data, exhausted raw/
candidate/result/pixel budgets and native exceptions emit no current result;
the common executor logs the error and supports recovery. Defaults bound raw
records to 10,000 and candidates/results to 2,048, with a maximum of 4,096
candidates (bounded quadratic work). Excess work rejects instead of truncating.
Failed-input admission can retain prior history and emit nothing. Independent
instances are stateless and can participate in parallel branch scheduling.

[Thirteen original projects](<../examples/opencv tools/soft_nms/README.md>)
compare soft/hard/disabled suppression, fractional/border boxes, classes, source
selections, high depth and count judgments using a known image-dependent ONNX head.
They qualify postprocessing/dataflow behavior, not learned recognition accuracy.
