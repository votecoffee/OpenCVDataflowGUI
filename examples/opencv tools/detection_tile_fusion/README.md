# Detection fusion across overlapping tiles

Two overlapping crops of a credited real photograph run through independent
YOLOX-S detectors. Join parallel branches groups their ticketed box collections;
**Fuse tile detections** maps them to the original image and suppresses duplicates.
The original image remains unchanged. Drawable result graphics feed an overlay
block and the filmstrip. No plugin is required.

## Prepare and run

From the repository root, using the application's Python environment:

```powershell
python "examples/opencv tools/detection_tile_fusion/prepare_tile_model.py"
```

This explicitly downloads and verifies the pinned 35,858,002-byte Apache 2.0
YOLOX-S model, then seals its model/settings bundle under ignored `.local`.
`--model <existing.onnx>` reuses identical weights. Different existing files are
preserved. Opening and running a project never downloads anything. No training
framework, ONNX package or ONNX Runtime is required.

Open [coffee.json](coffee.json) and Run. Both crops detect the cup; the fusion
block returns **one** source-image cup, roughly score **0.53** on the qualified
Classic / OpenCV / CPU path. Inspect its report for source frames, tile mappings
and filtering counts. Enable **Include candidate details** to inspect
each mapping/filter/suppression decision. Lean omits optional result graphics.

| Project | Expected run | Demonstrates |
| --- | --- | --- |
| [coffee.json](coffee.json) | Pass, 1 cup | Two detections become one on the reference grid. |
| [duplicates.json](duplicates.json) | Pass, 2 boxes | Fusion NMS off: validation finds 1 true positive and 1 duplicate false positive. |
| [resized.json](resized.json) | Pass, 1 cup | Right crop is halved before inference; edge coordinates scale back correctly. |
| [selected_out.json](selected_out.json) | Fail, 0 | Reference-grid region excludes the cup center. |
| [count_fail.json](count_fail.json) | Fail, 1 | Minimum count 2 fails while preserving the detected geometry. |
| [failed_tile.json](failed_tile.json) | Fail, 1 | Right decoder requires 2 cups; fusion retains its observations and reports a failed tile. |
| [empty.json](empty.json) | Fail, 0 | Black negative gives valid empty collections and a count failure. |
| [missing_tile.json](missing_tile.json) | Error, no event | Intentionally requires 3 tiles from a supplied group of 2. |

The crops are `[80,0,440,380]` and `[140,0,420,400]` (XYWH); both contain the
hand-marked cup. The right detector's unscaled observation underestimates its
height; the higher-scoring left observation wins ordinary hard NMS. Fusion does
not reconstruct truncated objects or promise accuracy improvements. Separate
exact geometry tests cover partial observations at internal seams, edge filtering,
outer-boundary exemptions, flips and right-angle rotations.

Both decoders use minimum count **0**, so an empty tile is a valid observation.
Only fusion normally enforces overall object count. A tile execution exception
prevents the join from completing; it does not create a passing empty detection.
An explicitly failed supplied tile produces the separate `tile_detections.failed_tile`
reason. Input failure policies can gate the branches if that is preferred.

## Ground truth and scope

Each graph has a **cup** validation set with independent source annotation
`[170,18,245,292]` and IoU 0.6. Coffee, resized, count-fail, failed-tile and empty
controls pass their declared validation expectations. Duplicates and selection
exclusion intentionally fail detector validation. The missing-tile error retains
the missed ground-truth cup and fails validation. This is one photograph and a
black negative, not an accuracy benchmark or industrial-parts qualification.

```powershell
python -m opencv_dataflow.assistant_cli run-validation "examples/opencv tools/detection_tile_fusion/coffee.json" --set cup --report tile-report.json --summary
```

Use a fresh report filename. Save As copies the sealed model bundle, image assets
and declared attribution/license files. Graphs explicitly use two branch workers;
focused checks compare serial and parallel results, independent native preprocessing
and box transforms, Debug/Lean, XML and cold-worker teardown.

Weights and image credits are in [SOURCES.md](SOURCES.md) and on every graph.
See [the block contract](../../../docs/tile-detection-fusion.md) for supported
coordinate maps, seam policies and resource limits.
