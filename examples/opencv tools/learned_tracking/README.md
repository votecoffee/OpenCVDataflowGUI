# Learned video tracking with native ViT

These nine projects include the small pinned Apache ViTTrack ONNX, exact notice
and attributed public T-LESS texture. Open a project and choose **Run all** to
replay its annotated clip. No plugin, model download or training package is needed.
See [the block contract](../../../docs/learned-tracking.md) and [sources](SOURCES.md).

| Project | Purpose |
| --- | --- |
| [translation](translation.json) | Seven-frame known translation; warm-up then current boxes/poses |
| [loss_expiry](loss_expiry.json) | Blank frames, native score loss and expiry; no automatic reacquisition |
| [detector_reseed](detector_reseed.json) | Template detector supplies explicit seeds and creates a fresh track after loss |
| [region_seed](region_seed.json) | Rectangle selection supplies the initial seed and gates accepted centers |
| [score_alarm](score_alarm.json) | Deliberately high threshold rejects even visible-target updates |
| [jump_alarm](jump_alarm.json) | Motion limit rejects a candidate despite a high native score |
| [resource_limit](resource_limit.json) | Deliberately undersized crop budget causes a contained configuration error |
| [new_engine](new_engine.json) | Same pinned model with explicit New/CPU import |
| [grayscale](grayscale.json) | Native tracking receives canonical BGR from grayscale input |

The first activation initializes a template. With warmup=3, geometry is withheld
for the first two activations. **Ready** indicates current geometry is available.
**Image** always carries the current original input. **Tracking score** is a native
window-weighted response, not a probability; it is absent before an update.
Lost/expired tracks do not emit current Bounds/Pose/Tracked region. Status retains
last accepted bounds as history. Drawable status/box graphics appear on the current
image; optional raster Overlay and model Receipt connectors are hidden initially.
Lean omits graphics while preserving the same native state/decisions.

After expiry, a fresh detector Seed or Reset is required. Initial configured bounds
are not silently used to find a lost object. The detector example demonstrates
re-initialization and a new track ID. Pose angle follows image axes and does not
measure part orientation.

These clips are artificial translations of one public image, not acquired video
or occlusion ground truth. The 43x43 evaluation ROI moves by (2,1) pixels/frame.
[Direct native results](benchmark.json) give minimum box IoU 0.93284, maximum
center error 1.11804 pixels and blank-frame score 0.09838 below the 0.2 threshold.
Run the independent benchmark from the repository root:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/learned_tracking/benchmark.py"
```

`prepare_model.py` explicitly re-verifies/reacquires the shipped pinned weight file;
it is not needed for ordinary use. `build_example.py` regenerates exact assets and
configuration-specific sealed bundles without changing the model. Existing
differing images/weights are preserved. Save As carries exact weights/notices;
JSON/XML reopen starts a fresh tracker rather than retaining a live native handle.
To change sealed settings, export a new contract through the model API.
