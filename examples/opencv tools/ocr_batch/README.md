# OCR regions in native batches

Open `dynamic_batch.json` and run once. Four source regions retain their order
and produce `ABA`, `BA`, `AAA`, `BBB` through genuine native batches of three
and one. The source image remains unchanged; recognized strings and region
outlines reach the filmstrip through Result graphics. Select the OCR block to
inspect each region's text, confidence, character spans and status.

| Project | Expected result |
| --- | --- |
| `dynamic_batch.json` | Pass; four ordered reads in native calls N=3 and N=1 |
| `padded_tail.json` | Pass; N=3/N=3 with two discarded padding rows, still four reads |
| `blank_failure.json` | Fail; one blank row retained among four good reads |
| `minimum_reads.json` | Pass; explicitly require four good reads despite a blank sibling |
| `confidence_gate.json` | Fail; keep raw texts/scores but accept none at the very high limit |
| `selected_roi.json` | Pass; two reads, with skipped rows retaining their original indices |
| `empty_selection.json` | Fail; no inference, no model load, current empty/skip evidence |
| `native_serial.json` | Pass; pinned CRNN reads `lot2048`, `pnax12`, `pass` in three N=1 calls |

The first seven projects are self-contained and need no download or plugin.
Their tiny original MIT models decode known intensity patterns to verify
batching and CTC bookkeeping. They do not demonstrate learned OCR accuracy.

For the optional printed-label project, prepare the unchanged Apache-2.0 model:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/ocr_batch/prepare_model.py"
```

This explicit step downloads and verifies 33.8 MB, then seals model/settings
under ignored `.local/crnn_en`. `--model <path>` reuses the identical weights.
Opening or running a saved program never downloads anything. This pinned
export supports N=1; its batch size is deliberately one. Larger native batches
require a compatible export and explicit input/output layout settings.

Use **Regions** to connect `ocr_regions_v1` from Text regions from probability
map. Without that connector, **Quadrilaterals** can supply source quads; an empty
list uses the whole image or selection bounds. Mask/Region filters centers and
fills excluded pixels with Background. Returned indices always refer to original
region order. Raw text, confidence and failure status stay visible even when
the aggregate fails; the Text connector contains accepted reads only.

Confidence is the geometric mean of the highest chosen probability in each
collapsed nonblank run. It is a greedy-path score, not calibrated string
correctness or the complete CTC posterior. Keep process-specific acceptance
checks separate. Debug adds drawable per-read colors/text; Lean retains the
same numeric reads without graphics.

`build_example.py` regenerates graphs, images and arithmetic ONNX models in a
fresh folder with the optional ONNX authoring package. Existing different assets
are preserved. Save As copies a sealed native model and notices and rebases
source-image references; copy the whole example folder for self-contained use.
See [SOURCES.md](SOURCES.md), [the upstream license](MODEL-LICENSE.txt), and
[the full batch OCR contract](../../../docs/batch-ocr.md).
