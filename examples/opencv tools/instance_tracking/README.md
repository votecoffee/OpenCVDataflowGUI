# Instance masks with temporal IDs

Open `crossing.json`, choose **Run all**, then inspect **Track instance masks**.
IDs remain stable as same-class red/blue parts cross. The tracked collection can
be passed to **Select instance mask**, sorting/filtering and downstream inspections.
The filmstrip receives current masks with track labels drawn on the source image.

| Project | Demonstration |
| --- | --- |
| `crossing.json` | Nine frames; brief full occlusion retains the missing track and its later ID. |
| `occlusion.json` | Two missing blue observations; returning pixels regain the same track. |
| `expiry.json` | Eleven frames; a longer absence expires blue and creates a new ID on re-entry. |
| `transparent_overlap.json` | Independent overlapping masks; identical ambiguous masks are withheld explicitly. |
| `confirmation.json` | Two observations required before current masks are emitted. |
| `high_depth.json` | Original uint16 image retained; inference/appearance use uint8 working data. |
| `gray_appearance.json` | Explicit gray histogram alternative to masked HSV appearance. |
| `decode_budget.json` | Deliberate rasterization-budget error with no partial tracking outputs. |

The report separates current observations from missing-track predictions and lists
associations, ambiguities and expiry. Missing tracks have no emitted mask, even
while their predicted centers remain in Report. Generations distinguish numeric IDs
after reset. Save/reopen starts with no live tracks.

These original synthetic videos and the small channel-mask ONNX model exercise
the reusable segmentation/association workflow. They do not qualify a pretrained
segmentation backbone or physical identity under general occlusion. Expiry has one
intentional ID switch; confirmation and ambiguity each withhold two visible masks.

`truth.json` records independent visible masks and seven asset hashes. `benchmark.py`
checks direct native masks and independently scores workflow masks/ID histories;
`benchmark.json` records every frame and intentional failure. The builder regenerates
into an empty folder and preserves existing assets; optional `onnx` is needed only
to rebuild, not to run the saved projects. See [sources](SOURCES.md) and
[contracts](../../../docs/instance-tracking.md).
