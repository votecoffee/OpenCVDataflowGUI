# Classifying parts in pose-aligned crops

Start with `all_known.json`: six parts at different angles should pass. Then open
`parallel_inspection.json`: five textured parts pass and an unmarked part fails on
low confidence. Blob measurements supply the poses; the reusable inspection
ToolBlock crops each part, converts an inference copy, runs a sealed ONNX model
and reports class/confidence. Enter the inspection block to inspect the selected
part's crop, source label/footprint and intermediate steps.

| Project | Demonstration |
| --- | --- |
| `all_known.json` | Six textured parts at different angles; all pass. |
| `parallel_inspection.json` | Three workers; five known textures pass, one unmarked part fails. |
| `serial_inspection.json` | One worker, same crops/classes/outcomes as parallel processing. |
| `accepted_horizontal.json` | Only horizontal texture accepted: three pass, three fail. |
| `no_alignment.json` | Fixed image axes instead of measured axes: all six fail the confidence criterion. |
| `high_depth.json` | Same inspection from uint16; returned crop depth and mappings are retained. |
| `single_part.json` | One measured part and one child inspection ticket. |
| `selected_parts.json` | A left-side region selects two measured parts; both pass. |
| `empty.json` | No parts: ordinary empty inspection failure, no fabricated jobs. |
| `coverage_pad.json` | Partial pose crop at an image corner: about 40.09% measured support, valid failure. |
| `coverage_strict.json` | Same unsupported crop is rejected with no crop output event. |

The parent filmstrip shows status/part IDs. The result's ordered images group
contains rectified crop pixels with child tickets, parent-ticket references and
source coordinate maps. Existing parallel-inspection aggregate and early-stop
controls remain available. These projects wait for all jobs to finish.

All data/model/graphs are original MIT fixtures. Twenty-four separate noisy windows
train a binary linear SVM, whose weights are exported as a 4,313-byte ONNX model.
The illustrative softmax scores are not calibrated probabilities or evidence of
industrial accuracy. The measured rectangular axis has a 180-degree ambiguity;
these two stripe classes are invariant to it. See [sources](SOURCES.md),
[contract](../../../docs/pose-crop-classification.md), `truth.json` and `benchmark.json`.

`benchmark.py` imports no application processing implementation. It independently
checks component moments, bilinear samples and coefficient/native inference.
`build_example.py` regenerates into an empty destination and refuses to replace
existing assets. Rebuilding uses the optional `onnx` development dependency;
running the supplied projects only needs the ordinary application installation.
Attribution comments are included in every graph. No customer captures, model
fetches, equipment or external plugins are required.
