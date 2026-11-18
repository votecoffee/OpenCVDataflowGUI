# Cascade detection with HAAR and LBP

Open `portrait.json` and run it to see a native face box and weight on the NASA
portrait. `two_faces.json` searches a composed view at two sizes. The filmstrip
uses drawable result overlays; double-click a result to inspect it. All files are
local. See [sources and rights](SOURCES.md) and the [block contract](../../../docs/cascade-detection.md).

| Project | Behavior |
| --- | --- |
| `portrait.json` | HAAR face at [176,65,97,97], native weight about 5.82965. |
| `neighbors.json` | Same box, with 37 grouped positive-window neighbors. |
| `boxes.json` | Same box, intentionally unscored. |
| `lbp.json` | LBP model returns [170,63,107,107], with different stage evidence. |
| `two_faces.json` | Original portrait plus a half-sized copy: two detections. |
| `selection.json` | The same two-size image, with a left-side region retaining one face. |
| `high_depth.json` | uint16 image retains original depth/pixels and the same box. |
| `equalized.json` | Optional working histogram equalization, with original output pixels. |
| `raw_windows.json` | Zero neighbors returns 40 windows, including extra boxes and negative weights. |
| `single_scale.json` | Requested 97×97 size selects native nearest 100×100 scale; strict size is off. |
| `strict_size.json` | Same native fallback, filtered to the exact requested size: valid empty failure. |
| `weight_fail.json` | Weight minimum 20 yields a valid empty inspection failure. |
| `count_fail.json` | One detected face fails the configured minimum count two. |
| `blank.json` | Blank image yields a valid empty inspection failure. |
| `bad_pin.json` | Incorrect model SHA-256 is rejected with no current detector output. |

Weights and grouped neighbor counts are different native outputs; neither is a
probability. Lower grouping thresholds can increase false detections. The coarse
manual face rectangle is a smoke-test reference, not a face-localization benchmark
or evidence of industrial accuracy. This also does not identify a person.

`truth.json` pins four PNGs/two XML models and records coarse reference regions.
`benchmark.py` independently calls native detection and filters boxes; it imports
no application processing code. `benchmark.json` records its outputs. The builder
can regenerate assets/projects into an empty destination using this folder's
`assets` as inputs; it refuses to replace existing assets. Attribution is included
in every graph. No plugin, equipment or model download is needed to run them.
