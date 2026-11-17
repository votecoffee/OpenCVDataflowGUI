# Object-detection fusion across image tiles

`ml.fuse_tile_detections` consumes a **reference image** and a flat `FlowGroup` of
`vision_collection_v1` box collections. Use crop/resize, DNN inference and detection
decoding on each branch, then Join parallel branches. Existing image groups and
ToolBlocks can supply the same contract. This block does not extract tiles or
invoke models. Find it under **OpenCV → Deep Neural Networks (dnn) → Object detection**.

## Identity and geometry

The reference and every tile must have the same known spatial root ID, root
dimensions and acquisition ticket. Groups have 1–64 distinct nonempty tile labels.
`expected_tiles` optionally enforces supplied group cardinality; it does not prove
complete source-image coverage. An exception in a branch prevents a normal join
from completing. Different known recipe revisions, contradictory collection/wire
frames or pyramids, unknown roots/tickets and nonzero collection origins reject.

Spatial transforms map **pixel centers**, whereas detection rectangles use
continuous **pixel edges**. The relative map is `inverse(reference_to_root) @
tile_to_root`. Its edge translation is adjusted by `0.5 - linear @ [0.5,0.5]`
before mapping four corners. This preserves exact crop/resize/flip/right-angle
rectangles, including anisotropic scales. Arbitrary relative rotations, shear
and perspective reject because their mapped rectangles would overstate measured
geometry. A transformed reference is supported when the *relative* map satisfies
these constraints; absolute root coordinates alone do not determine the result.

The default box source is each local XYWH `geometry.bbox`. The explicit
**unclipped** mode requires `geometry.unclipped_xyxy` on every detection. A border
policy clips, discards or retains boxes crossing the reference image. Fully
outside rectangles are discarded. Optional mask and region inputs select each
unclipped mapped center on the reference grid, without cropping inference.

**Drop internal edges** removes observations touching an internal tile edge,
optionally within a local-pixel margin, before suppression. Reference outer edges
are exempt. This can favor complete observations in overlapping tiles, but needs
sufficient overlap and may remove all observations of a boundary object. It does
not infer missing foreground, join fragments, average boxes or estimate occlusion.

## Selection, suppression and evidence

Boxes require finite positive bounded geometry, nonnegative integer class IDs,
consistent labels and scores in `[0,1]`. Oriented measurements are unsupported.
Class filtering and a strict **greater-than** confidence gate precede native
`cv2.dnn.NMSBoxes`. Scores and threshold are converted to float32, matching native
comparison precision. NMS is class-aware by default; class-agnostic mode is
explicit. IoU, adaptive `eta` and `top_k` are exposed. `top_k` applies separately
per class, or globally in class-agnostic mode. NMS compares mapped **unclipped**
geometry, even if final reference output boxes are clipped.

Candidates are ordered by descending score, then supplied tile order and source
record order. NMS-off preserves this order and still applies confidence, class,
selection, edge and border gates. Result IDs are newly assigned in that order.
The report distinguishes native `nms_or_top_k` rejection from other filtering;
it does not fabricate a suppressed-to-winner association. Reordering exactly tied
observations can change the selected source provenance.

Outputs include unchanged reference pixels, a source-grid box collection, count
and a JSON-safe report. Each retained item has fresh reference-grid measurements
and `provenance` containing its tile label/index, selected box source and a copy
of the **original local detection**. Original local values are never relabeled as
reference measurements. Orientation remains unmeasured. The report retains each
tile's mapping, spatial/pyramid metadata and failed-input flag. The reference
ticket, timing and grid metadata travel with outputs; a common known recipe is
also retained. Candidate detail reporting is optional and off by default.

Count limits create an ordinary `tile_detections.count` failure. Any supplied
failed tile independently adds `tile_detections.failed_tile`; available geometry
is retained. Default input policies continue failed inputs. Use `reject_failed`
to suppress activation instead. Malformed data, identity conflicts, budget excess
and native exceptions use normal block error containment and emit no current
output event; a later valid activation can recover.

## Bounds and qualification

Reference images are limited to 16,384 pixels on each axis and a configurable
pixel budget. Raw detections, selected NMS candidates and retained results have
separate rejecting budgets; nothing silently truncates. The default candidate
limit is 4,096 to bound quadratic native suppression. Optional graphics are
drawable labels/outlines; Lean or per-block graphics settings skip them. Primary
pixels and input collections remain unchanged.

[Eight runnable examples](<../examples/opencv tools/detection_tile_fusion/README.md>)
use pinned Apache 2.0 YOLOX-S on overlapping crops of one credited CC0 photograph,
a resized branch and a black negative. An independent native preprocessing/grid
reference checks scored source boxes. Hand-marked truth, duplicate false positives,
failed/count/selection/missing-tile controls, exact seam/scale/flip/right-angle
units, malformed frames, budgets, exception recovery, serial/parallel parity,
Debug/Lean, XML, portable assets and a cold CPU worker qualify this axis-aligned
baseline. General object reconstruction, weighted/soft fusion, arbitrary
homographies, large-dataset accuracy and accelerator execution remain unqualified.

Native reference: [OpenCV 5.0.0 NMS implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/nms.cpp).
