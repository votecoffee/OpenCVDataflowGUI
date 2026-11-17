# Reusable palettes and image descriptor clusters

Open any project here with the application. These original synthetic controls
need only the base OpenCV installation: no downloaded models or optional ML
framework. Palette training and held-out images have different arrangements;
HOG training and query images use separate stripe positions.

| Project | Expected behavior |
| --- | --- |
| `train_palette.json` | Fit three reference colors and save a new `trained-palette.json`. |
| `palette_ok.json` | Load the bundled BGR palette; recognize the rearranged colors with fixed IDs 2, 0, 1. |
| `palette_defect.json` | Fail: the 16×20 white defect has 320 unknown pixels; show its drawable outline in the filmstrip. |
| `palette_rgb16.json` | Same labels as the good case, with unchanged 16-bit RGB primary pixels. |
| `palette_lab.json` | Apply a saved normalized-Lab palette to the good image. |
| `palette_selection.json` | Pass: a rectangle excludes the deliberate defect; only the selected pixels are inspected. |
| `train_vocabulary.json` | Run All over six images, batch their native HOG descriptors and save a new `trained-vocabulary.json`. |
| `vocabulary_query.json` | Run All over two held-out stripe images; inspect IDs, distances and histogram outputs in Results. |
| `vocabulary_wrong_contract.json` | Deliberate error: signed HOG gradients change preprocessing at the same vector length; emit no predictions. |

Training preserves existing files. Before training again, choose a new save path.
To test a newly trained model, change the loader in the corresponding prediction
project to that file. The bundled asset models remain unchanged. Save As carries
the selected model with the project.

**Max distance** flags unfamiliar samples; **Accepted clusters** restricts known
IDs; **Max unknown fraction** controls the inspection judgment. The histogram
uses accepted counts divided by all selected samples, so unknown pixels reduce
its sum. Distances are working-space Euclidean distances, not probabilities.
Lab uses scaled channels and is not Delta E. IDs are meaningful only within the
same model revision. Color sampling is evenly spaced, not a guarantee of rare
color coverage. These controlled fixtures do not establish production accuracy.

The palette Image output is the original image. Quantized and Cluster map are
separate outputs. Result graphics contain drawable objects; the example composes
them before choosing the filmstrip image. Lean omits optional drawings/maps.

`truth.json` pins every source image/model file, the independently specified color
layout and defect rectangle. `build_example.py` rebuilds into a new folder and
refuses to overwrite existing assets. All projects include an attribution comment.
See [SOURCES](SOURCES.md) and [full contracts](../../../docs/reusable-clusters.md).
