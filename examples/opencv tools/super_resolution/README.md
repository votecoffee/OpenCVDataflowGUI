# Learned image super-resolution

Open a saved project in this folder and Run once. Each starts with a known
256x192 source, downsamples it to 128x96, then compares learned x2 and cubic
reconstruction against the original. Click the two error blocks for MAE/maps;
the learned block offers reconstruction, baseline and processed-selection images.
The small unchanged Apache-2.0 FSRCNN model is included; no preparation,
TensorFlow installation or download is required.

| Project | Demonstration |
| --- | --- |
| `geometry.json` | Circle, slot, bead/gap and cracks with independently specified native dimensions. |
| `color.json` | Known BGR channel values and color reconstruction. |
| `alpha.json` | BGRA: infer only color; interpolate alpha with the baseline. |
| `gray16.json` | Explicit 8-bit working quantization and uint16 output; no claim of added precision. |
| `selected_region.json` | Learn inside a circular source region; retain cubic interpolation elsewhere. |
| `empty_region.json` | Off-image selection emits only the baseline, with no native model call. |
| `fidelity_alarm.json` | Successful inference with an intentionally failed zero-error fidelity check. |
| `noise.json` | Seeded noise and the same geometry; compare smoothing and retained detail. |

On the clean control, learned reconstruction has lower global MAE than cubic
(1.1711 versus 1.7218), but both lose the one-pixel crack and shrink the
thresholded circle from 73x73 to 71x71. Learned reconstruction erases 27 dark
detail pixels versus cubic's 26. The original flat region is exactly uniform;
the learned result changes 30 of its 896 samples. On noisy input the learned
flat-region deviation is 3.0697, compared with linear's 2.3343. These controlled
results do not establish accuracy on real inspection images.

`truth.json` retains exact asset identities, dimensions, ROIs and thresholds.
`benchmark.json` contains independent direct-native measurements for learned,
linear, cubic and Lanczos results. The noise benchmark compares to the clean
known reference; the noise project's live MAE blocks compare to its noisy source
to show reconstruction error for that acquisition. Neither silently replaces
reference data with a generated image. Nominal coordinate mapping is preserved
through reduction/upscaling; actual learned feature alignment remains uncalibrated.

Run `build_example.py` in the app's development environment to regenerate
original fixtures/projects. `benchmark.py` independently measures the native
model. Both preserve differing retained assets/results for review. Tests are
in `tests/test_super_resolution.py` and `tests/test_super_resolution_examples.py`.
See the [block contract](../../../docs/super-resolution.md),
[sources/attribution](SOURCES.md) and [upstream license](MODEL-LICENSE.txt).
Each saved graph includes an attribution comment. All images are original
synthetic tool controls; no customer images or private assets are present.
