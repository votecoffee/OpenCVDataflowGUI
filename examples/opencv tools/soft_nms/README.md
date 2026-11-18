# Soft-NMS on overlapping proposals

Open [linear.json](linear.json) and press **Run once**. Select **Suppress** to
inspect its collection, ranked input indices, updated scores and candidate report.
The overlay and filmstrip show retained boxes in green and removed candidates in
red. All images and the small ONNX model are included; no downloads are needed.

The original synthetic image and image-dependent proposal head yield six known
boxes: an overlapping pair, a neighboring part, an overlapping different class,
a border-crossing box and a weak proposal. These are controlled postprocessing
inputs, not a learned detector. The decoder leaves suppression off so each method
receives all six proposals.

| Project | Purpose |
| --- | --- |
| [linear.json](linear.json) | Linear decay retains the shifted proposal at a lower score and changes its rank. |
| [gaussian.json](gaussian.json) | Gaussian decay uses sigma; the native method ignores the IoU cutoff. |
| [hard.json](hard.json) | Same integer boxes with native hard suppression: five retained versus six with the default soft controls. |
| [none.json](none.json) | No native suppression; inspect original ranked candidates. |
| [agnostic.json](agnostic.json) | Let the overlapping second class participate in the same decay group. |
| [top_k.json](top_k.json) | Keep at most two per class, then rank the combined results. |
| [selected.json](selected.json) | A source-space region selects the neighboring and weak proposals by center. |
| [empty.json](empty.json) | Deliberate count failure after a 0.99 score threshold removes everything. |
| [count_failure.json](count_failure.json) | Deliberate count failure: seven required, only six retained. |
| [high_depth.json](high_depth.json) | uint16 input and matching normalization; original pixels/depth remain unchanged. |
| [nearest.json](nearest.json) | Nearest-edge rounding versus Cover, preserving fractional output geometry. |
| [visible_boxes.json](visible_boxes.json) | Explicitly use clipped bbox geometry instead of full border-crossing boxes for overlap. |
| [details_off.json](details_off.json) | Skip the optional candidate trace while retaining aligned indices/scores. |

Input indices refer to the decoder's already score-sorted collection; item IDs
still identify original model rows. Each retained record preserves its original
score and the integer rectangle used by native suppression. Removed final scores
are unavailable, not zero. Threshold equality and Top k semantics are documented
in [the block contract](../../../docs/soft-nms.md).

Debug draws the overlays; Lean retains numerical results while skipping optional
graphics. Save As and XML round trips preserve parameters and image/model paths.
`truth.json` pins both assets and the independent proposal geometry, score biases
and class IDs. `build_example.py` regenerates everything in a new destination using
the development `onnx` package, and refuses to replace existing assets.

[Sources and licenses](SOURCES.md).
