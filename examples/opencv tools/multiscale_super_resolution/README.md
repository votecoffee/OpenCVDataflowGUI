# One inference, several learned image scales

Open [geometry.json](geometry.json) and press **Run once**. **Learned** produces
2×, 4× and 8× reconstructions from one named native forward. Select it to inspect
the image group, scale mappings and model hash. Its default preview/filmstrip is
8×; expose Image x2/x4/x8 in the sidebar to wire individual scales. The geometry
project already sends each scale to threshold/blob tools.

The model, fixtures, attribution and license are included. No downloads or
TensorFlow installation are needed. The implementation uses named native DNN
outputs because this wheel's `upsampleMultioutput` binding does not deliver images
to Python. See [the block contract](../../../docs/multiscale-super-resolution.md).

| Project | What to inspect |
| --- | --- |
| [geometry.json](geometry.json) | Three scale branches preserve the source coordinate frame; disk centers map back within one reference pixel. |
| [color.json](color.json) | Native luminance inference and chroma reconstruction. |
| [alpha.json](alpha.json) | Four-channel input retains interpolated alpha. |
| [gray16.json](gray16.json) | Explicit 8-bit working conversion, restored uint16 output. |
| [float.json](float.json) | Normalized float input/output with explicit working quantization. |
| [selected_region.json](selected_region.json) | Source-space left-half selection; baseline pixels outside it at every scale. |
| [empty_region.json](empty_region.json) | Baselines only: zero native calls. |
| [reordered.json](reordered.json) | Group order 8×/2×/4× and explicit 2× preview. |
| [subset.json](subset.json) | Only 2×/4×, with 4× as the largest preview. No 8× event. |
| [wrong_scale.json](wrong_scale.json) | Expected error: swapped node names disagree with declared output dimensions. |
| [budget_error.json](budget_error.json) | Expected error before inference: aggregate output budget is too small. |
| [noise.json](noise.json) | Seeded acquisition noise and learned reconstruction. |
| [coffee.json](coffee.json) | Public CC0 photo through the same scale pipeline. |

The synthetic native-resolution reference has independently specified geometry,
including a disk, slot, bead gap and fine cracks. `benchmark.py` separately calls
the native model and compares learned, linear and cubic outputs with that truth
at each scale. `benchmark.json` records pixel errors, mapped centers, erased
detail and false detail. It does not substitute reconstructed output for truth.

Clean learned disk errors are about 0.59/0.16/0.23 reference pixels at 2×/4×/8×;
the 8× output loses 230 dark crack-detail pixels in the defined ROI. This is a
controlled reconstruction demonstration, not calibrated metrology or proof that
super-resolution improves inspection accuracy. The display label identifies the
nominal reconstruction and is hidden with optional graphics in Lean mode.

`truth.json` pins six bundled image/model files. `build_example.py` regenerates
public assets/projects into a new destination, verifies the exact model and
refuses an existing asset folder. It reuses the original fixture generator from
the sibling single-image example. Save As copies the model and source/license
documents, while preserving the model bytes.

[Sources and licenses](SOURCES.md).
