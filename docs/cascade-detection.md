# Cascade object detection

`ml.cascade_detect` loads a local modern OpenCV HAAR or LBP BOOST XML model and
searches the current image. It supports independent branch execution, last-input
preview, shared timing/diagnostics, ticket outcomes and drawable box/label results.
Try the [fifteen public projects](<../examples/opencv tools/cascade_detection/README.md>).

## Models and controls

The model path is a sealed project asset: Save As carries its exact bytes.
An optional 64-digit SHA-256 pin rejects replaced or edited files. An unpinned
model is read and hashed on every invocation, so changing a local file cannot
leave a stale native classifier. The model report records bytes, digest, OpenCV
version, original window, feature type, stages, nodes and feature count. Validation
and project loading do not open the file or initialize native processing.

Only modern XML with BOOST/HAAR or BOOST/LBP structure is supported. Legacy cascade
conversion, YAML and classifier training are separate workflows. Old-format flags
are omitted because OpenCV ignores them for modern cascades. Native parsing uses
FileStorage memory from the same bounded bytes that were hashed, avoiding a second
path read. The classifier is invocation-local and released after processing.

Expose `scale_factor`, `min_neighbors`, `min_size`, `max_size`, and optional working
histogram equalization. Minimum/maximum sizes are [width,height]; [0,0] means native
minimum/image maximum. The scale factor is restricted to 1.01..4 to bound the scale
inventory. Zero neighbors exposes ungrouped positive windows and may return many
extra or overlapping boxes.

OpenCV chooses from precomputed rounded scales. When no scale matches the size
range, the native implementation can choose the nearest scale even outside the
requested range. The report identifies this fallback and its effective windows.
`strict_size=True` then filters the returned visible boxes to the requested sizes.
Turn it off to inspect native behavior, including nearest-scale single-size
searches. Grouped/clipped boxes can also fall outside requested dimensions.

## Three kinds of result evidence

| Mode | Native method | Collection evidence |
| --- | --- | --- |
| `weights` | detectMultiScale3 with outputRejectLevels | Final-stage `native_weight`, `reject_level`, score kind `cascade_stage_weight` |
| `neighbors` | detectMultiScale2 | Grouped positive-window `neighbor_count`, score kind `cascade_group_neighbors` |
| `boxes` | detectMultiScale | Unscored boxes with `score=None` |

Neither stage weights nor neighbor counts are probabilities. Stage weights can be
negative even for a returned positive window. `reject_level` describes native
stage evidence on returned detections; the output is not a collection of all
rejected scan windows. Weight filtering is optional and inclusive, enabled
separately so a default zero threshold does not discard negative weights. It
requires weights mode. Native grouping controls duplicate support; there is no
additional hidden suppression or score calibration.

All modes return `vision_collection_v1` boxes with common size/position/area/aspect
metrics, label, original native index, and explicit unmeasured orientation. Stable
coordinate order supports downstream sorting/filtering. Minimum/maximum retained
counts yield an ordinary inspection judgment and `cascade.count` reason. A zero
maximum disables that inspection upper limit. Overall Result.score stays unset.

## Image, selection and execution behavior

The primary image output retains original pixels, depth and encoding. A fixed
uint8 gray working conversion supports uint8/uint16 and finite float [0,1] inputs;
there is no per-frame intensity normalization. Optional equalization changes only
the working image. Ticket, spatial, pyramid and custom payload remain associated
with current image coordinates. Region and mask intersect; transparent pixels
are excluded. Wired optional selection inputs require a current value.

Selection tests returned visible box centers, selected box coverage, or both.
Gradients/features still read the whole working image. Native clipping means
coverage describes the visible box rather than an unavailable unclipped footprint.
Drawings respect global/per-block graphics policies and can be composed onto a
filmstrip image with the ordinary overlay block. Exceptions remain contained;
failed-input gating emits nothing and retains previous processed-input history.

Image axes/pixels, model bytes/complexity, scale inventory, conservative pixel-start
window count, full-tree work and estimated workspace are bounded before native
model loading/search. Native candidate and retained-result limits reject excess
output without silently truncating it. Work/memory figures are estimates rather
than measured operation counts, peak RSS or time guarantees. The native search is
synchronous. CPU execution restores the calling thread's OpenCL preference on
success/error; normal scheduling prepares native lifetime before temporary workers
as described in [architecture](architecture.md#native-worker-startup).

## Qualification

OpenCV 4.13.0 XML models run under installed OpenCV 5.0.0. Both model files retain
source terms; the HAAR file includes its Intel license and the LBP model uses the
accompanying OpenCV Apache-2 license. A public-domain NASA portrait has one coarse
manual face ROI. A two-size composition has a second ROI derived by the known
half-scale/translation; those annotations predate example validation, without
being an independent localization benchmark. Both model boxes overlap the coarse
portrait ROI above 0.7 IoU; the two-face HAAR scene returns two boxes.

The portrait's native HAAR box is [176,65,97,97], with weight about 5.82965 and
37 grouped neighbors. LBP returns [170,63,107,107]. Ungrouped HAAR returns 40 boxes,
including extras with negative margins. Nearest-scale control requests 97×97,
selects native 100×100, and strict size filtering yields a valid empty failure.
These are pinned CPU smoke tests, not general face/industrial accuracy claims.

The independent benchmark imports no application processing code: it directly
calls native methods and separately filters scores/sizes/selection. Tests compare
boxes, native evidence, controls, Debug/Lean, high depth/payloads, strict failures,
asset regeneration, JSON/XML/Save As, input gating and cold Run All/parallel/preview
teardown. No hardware, accelerator, new training algorithm or runtime download is
required by these examples.
