# Detection tracking and appearance

Open a project and choose **Run All**. Actual OpenCV blob or YOLOX detections feed
Describe box appearance, Track image detections and final filmstrip overlays.
Green tracks are observed; amber tracks are predictions. Reset flow state before
replaying. No camera, controller or plugin is required.

| Project | TP / FP / FN | Identity switches / fragments | Purpose |
| --- | --- | --- | --- |
| [crossing.json](crossing.json) | 18 / 0 / 0 | 0 / 0 | Parts cross in image X at different Y positions. |
| [occluded_swap.json](occluded_swap.json) | 14 / 0 / 0 | 0 / 0 | Parts exchange positions during two hidden frames; appearance retains identity. |
| [motion_only.json](motion_only.json) | 14 / 0 / 0 | 2 / 0 | Same images, appearance off: predicted positions exchange identities. |
| [detection_gap.json](detection_gap.json) | 6 / 0 / 3 | 0 / 1 | A dim visible part falls below the threshold for three frames. |
| [expire_reentry.json](expire_reentry.json) | 6 / 0 / 3 | 1 / 1 | Same dim frames, shorter expiry: re-entry receives a new identity. |
| [yolox_coffee.json](yolox_coffee.json) | 5 / 0 / 0 | 0 / 0 | Learned detections on translated credited photographs, with one hidden frame. |

The first five projects run immediately. For YOLOX, prepare its pinned Apache 2.0
model explicitly from the repository root:

```powershell
python "examples/opencv tools/detection_tracking/prepare_tracking_model.py"
```

`--model <existing.onnx>` reuses identical weights. Preparation reuses the adjacent
tile-fusion example's checker and seals this folder's ignored local bundle.
No model is downloaded or converted when opening/running a graph. The unmodified
YOLOX-S model is 35,858,002 bytes, using Classic / OpenCV / CPU, float32 RGB 0..255
and centered 640×640 letterboxing. No training framework, ONNX package or ONNX
Runtime is required.

## Inspect appearance and history

Describe box appearance adds a named, normalized HSV hue/saturation histogram to
each detection. It samples an inset rectangular ROI on the **original color
image**, preserving records, pixels and ticket/grid metadata. The threshold
branch supplies detections, not histogram pixels. Mask/region and alpha can
restrict sampling; grayscale histograms are also available. These are color
descriptors, not learned re-identification or unique part IDs.

The tracker defaults to appearance **off** for existing programs. These examples
use **required**, except the deliberate motion-only control. Appearance adds an
association gate/cost; motion, class, scale and angle gates still apply. Inspect
source detection IDs, appearance availability/space, association distances,
ambiguity and expiry reasons. A retained appearance template is not fresh
evidence during prediction. Gate picking/inspection on observed/detector flags.

The hidden swap is deliberately ambiguous to a stationary motion model; it does
not measure motion while objects are invisible. Dim frames retain a visible part
and independent annotation, so detector misses remain in the score.

## Reproduce metrics

```powershell
python "examples/opencv tools/detection_tracking/evaluate_examples.py" occluded_swap --report tracking-report.json
```

Use a fresh report filename. `--profile lean` retains estimates while omitting
optional graphics. The evaluator verifies image hashes, explicitly runs every
frame, requires current tracker events and scores **observed output tracks only**
at source-grid IoU ≥ 0.6. Predicted counts are reported separately; hidden
predictions do not count as fresh detections. Truth never feeds the graph.

Class-compatible one-to-one global IoU assignment determines TP/FP/FN. Identity
switches compare the latest match even across absence. A fragment requires a
labeled miss between matches; absent truth alone does not create a fragment.
Long-expired returning parts honestly count as new identities. Reports include
per-frame matches, precision/recall, MOTA and mean matched IoU, without HOTA/IDF1.

All projects have exact regeneration and Debug/Lean checks. Native references,
normalization/gates, missing features, class IDs, frame/ticket guards, budgets,
exceptions/recovery, worker memory, XML and offscreen controls are tested.
Save As carries image/model assets and credits. The source-folder evaluation
script and annotations are a separate offline fixture workflow.

Original colored controls are MIT; coffee derivatives retain Rachel Michetti /
Pikolo Espresso Bar CC0 credit. [SOURCES.md](SOURCES.md) records changes and the
Apache model license. This is controlled software qualification, without a real
video benchmark, learned re-identification model or physical tracking claim.
See [tracking contracts](../../../docs/image-tracking.md).
