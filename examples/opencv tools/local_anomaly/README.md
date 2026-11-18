# Local image anomaly inspection

Open `normal.json`, then `defect.json`. Inspect Distance, Support, Mask and the
threshold-relative heatmap; the filmstrip receives the drawn result contours.
`train_then_inspect.json` shows three known-good images becoming a reference bank
inside the graph. Other projects load the sealed reference in `assets/reference.json`.

| Project | Demonstration |
| --- | --- |
| `normal.json` | Held-out surface passes with maximum distance about 0.00555. |
| `defect.json` | Dark 8×8 defect fails and localizes to exactly 64 source pixels. |
| `missing_texture.json` | An 8×8 removed texture fails and localizes to 64 pixels. |
| `weak_defect.json` | Deliberate false negative: a one-pixel defect falls below threshold. |
| `bright_shift.json` | Defect-free brightness shift rejects all 1024 pixels. |
| `selected_region.json` | A rectangle excludes the defect; 320 selected pixels pass. |
| `high_depth.json` | Original uint16 image is retained; inference uses a converted copy. |
| `allowed_area.json` | Inclusive allowance of 64 anomalous pixels accepts the dark defect. |
| `wrong_features.json` | Changed normalization is rejected against the trained contract. |
| `corrupt_reference.json` | Altered reference values are rejected by the content revision. |
| `train_then_inspect.json` | Train on three known-good images, then inspect a new image. |
| `sampled_reference.json` | Explicitly reduce 768 candidates to 16 evenly spaced references. |

The ONNX encoder represents each 2×2 cell by its four normalized pixel values.
The bank stores 768 local vectors from three training seeds. Query seeds and image
hashes are separate. No pretrained learned weights are included. These are focused
tool examples, not industrial accuracy or calibration evidence. A global reference
bank can miss rearrangements of known textures and is sensitive to lighting.

To save a new bank, connect the training Model to **Save local anomaly reference**
and choose a new filename. The save block preserves existing files. Load it with
**Load local anomaly reference**. Reassess thresholds after changing the encoder,
normalization, training images, distance metric or sampling.

`truth.json` pins the original assets. `benchmark.py` independently extracts pixel
patches, calls native ONNX and calculates distances/localization with NumPy.
`benchmark.json` includes the intentional missed defect and false rejection.
`build_example.py` regenerates into an empty directory and preserves existing
assets; rebuilding needs the optional `onnx` development package. Running the
saved projects does not. See [sources](SOURCES.md) and [contracts](../../../docs/local-anomaly.md).
