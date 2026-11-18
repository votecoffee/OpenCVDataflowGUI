# Continuous measurements from images

Start with `width_middle.json`, then `evaluation.json` for a three-image error
report. The model estimates a width proxy from average intensity. Under its
controlled lighting/background conditions it predicts the synthetic width labels;
`low_contrast.json` shows why those assumptions matter. Values have units, not
confidence scores. Source images are unchanged, with result text drawn separately.

| Project | Demonstration |
| --- | --- |
| `width_small.json` | Held-out 6.5-pixel bead; reference 0.65 mm. |
| `width_middle.json` | Held-out 13.25-pixel bead; reference 1.325 mm. |
| `width_large.json` | Held-out 21.75-pixel bead; reference 2.175 mm. |
| `high_depth.json` | Original uint16 display image; converted inference copy and unchanged output depth. |
| `brightness.json` | Select the second vector component and scale its value to gray levels. |
| `nchw_head.json` | Explicit [1,2,1,1] regression head. |
| `no_reference.json` | Measurement event without fabricated reference/residual events. |
| `low_contrast.json` | Dimmed bead underestimates width and fails the residual limit. |
| `range_fail.json` | Valid prediction fails a configured measurement range. |
| `wrong_layout.json` | Spatial-looking head rejected unless its NCHW scalar layout is declared. |
| `bad_model.json` | Altered model bytes rejected by the bundle checksum before inference. |
| `evaluation.json` | Three separate held-out images, referenced records and MAE/RMSE/bias/R² report. |
| `evaluation_lighting.json` | One dim image makes the aggregate MAE limit fail. |
| `evaluation_mixed.json` | Different model revisions rejected rather than pooled into one report. |
| `batch_rows.json` | Run all: collect three images, infer batches of two/one, unpack rows and evaluate all three. |

The reference scale is an assumed 0.1 mm/pixel, not camera calibration. The model
uses image mean rather than geometric edge measurement; changed lighting, bead
length or background can invalidate that relationship. Held-out synthetic MAE is
about 0.000594 mm. A dim image gives approximately -0.464 mm residual. Neither is
evidence of real machine accuracy or an automatic out-of-distribution detector.

Seven original images train a native linear EPS-SVR, then fixture generation
exports its coefficients to ONNX. Running the graphs uses existing ONNX inference
plus the new decoder/metrics, without a separate SVR-training plugin. `truth.json`
pins all 30 image/model/configuration artifacts. `benchmark.py` checks native SVR,
direct ONNX and independent numeric formulas; `benchmark.json` records the results.
The builder regenerates into an empty destination and preserves existing assets.
Rebuilding needs the optional `onnx` development package; the checked-in projects
run with the ordinary application installation. See [sources](SOURCES.md) and the
[contract](../../../docs/image-regression.md). Every graph includes attribution.
