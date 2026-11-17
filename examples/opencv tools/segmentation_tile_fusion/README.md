# Segmentation across overlapping tiles

Open `probability_mean.json` and run once. Two image crops execute their ONNX
branches independently, join their full class evidence, and fuse on the original
image. The red part crosses both overlap boundaries. Select **Fuse tile
segmentation** to inspect labels, mask, confidence, tile coverage, weight and
class statistics. The filmstrip receives the image with selected-region outlines.
A separate full-image branch provides a baseline in the same project.

No plugin, model download or optional runtime dependency is needed. These
original MIT synthetic images and tiny pixel-dependent ONNX equations are
tool-level controls, with no learned recognition or customer content. All input
images, model bytes, independent class truth and graphs are in this folder.
Credits are also embedded in each project's comment block. The model uses
`[background=1, red=4R, green=4G, blue=4B]` logits for normalized RGB.

| Project | Expected outcome |
| --- | --- |
| `probability_mean.json` | Pass; uniform probability fusion, 788 selected pixels |
| `feather.json` | Pass; feathered overlap, same controlled class labels |
| `label_vote.json` | Pass; weighted votes with explicit vote-fraction confidence |
| `resized_tile.json` | Pass; right tile at half resolution, restored to source coordinates |
| `selected_roi.json` | Pass; reference rectangle around the red part |
| `coverage_failure.json` | Fail; cropped gaps leave 1/3 of source unsupported |
| `failed_tile.json` | Fail; deliberately failed right tile remains visible in the aggregate |
| `area_failure.json` | Fail; selected fraction exceeds 1% |
| `empty.json` | Fail; black image has no selected part pixels |

Each graph retains its node layout, large fusion preview and two-worker branch
setting. Normal Debug/Lean settings control optional graphics. Changing overlap,
sampling resolution or model context may change real predictions even when the
coordinate mapping is correct. The resized fixture happens to retain the exact
labels because its synthetic boundaries align with the sample grid.

`assets/truth_labels_plus_one.png` stores independent class IDs plus one, so it
is lossless in PNG; subtract one to recover signed labels. `truth.json` records
asset hashes and class counts. `build_example.py` regenerates into a fresh
directory with the optional ONNX authoring package; it preserves differing
existing assets. Ordinary Save As preserves references to original external
inputs/model files; copy this whole folder to share a self-contained example.

See [the tile-fusion contract](../../../docs/tiled-segmentation.md) for valid
geometry, class evidence, metadata, judgments and resource limits. These
software fixtures establish pixel mapping and continuity, not production model
accuracy or hardware acceptance.
