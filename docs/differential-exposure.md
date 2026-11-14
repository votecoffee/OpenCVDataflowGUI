# Compare images after exposure compensation

**Compare exposure-adjusted images** (`inspect.differential_exposure`) fits an
aligned candidate/reference pair using `reference = gain * candidate + offset`.
It complements flat-field calibration, single-image illumination correction and
normalized correlation with explicit raw and compensated defect residuals.

Connect images that are already registered. This block performs no registration,
recovers no clipped scene information and does not calibrate physical radiance.
Choose stable fitting areas: a sufficiently broad defect can resemble illumination
and be learned away. Robust fitting helps sparse outliers; it cannot resolve that
ambiguity in every scene.

## Images and selections

Both inputs must have matching dimensions, depth and encoding: grayscale,
RGB/BGR or straight-alpha RGB(A)/BGR(A), using uint8, uint16, int16, float32 or
float64. Samples are compared in **raw intensity units**, with no implicit
full-scale conversion. A threshold of 15 means 15 native sample units for every
depth, including normalized floating data. Adjust thresholds for your data.
Alpha is preserved from the candidate and excluded from fitting/judgment.
Unassociate premultiplied alpha first. HSV/Lab and physical depth are rejected.

**Mask** and **Region** intersect to define inspection pixels. **Fit mask** and
**Fit region** independently intersect to select stable fitting pixels. If both
fitting inputs are absent, inspection pixels supply the fit. Fitting pixels may
lie outside inspection. All selections use the candidate pixel grid and the
usual mask/region coordinate checks. An empty inspection selection is invalid;
an empty or inadequate fit produces a failed inspection with explicit coverage.

**Pixel grid** explicitly assumes equal-index registration, allowing independent
reference captures with different frame IDs/tickets. **Same frame** requires
matching known root IDs, dimensions and pixel transforms. Matching metadata is
not a physical registration measurement. Outputs belong to the candidate ticket,
recipe and spatial/pyramid context; Info records the reference ticket and frame.

## Fitting controls

| Control | Meaning |
| --- | --- |
| Scope | Local: one weighted model in each centered box window. Global: one model per channel over the fit selection |
| Window | Odd local width/height, 3–255 pixels; unused in Global mode |
| Model | Gain/offset, or gain fixed at one for offset-only correction |
| Robust loss | Ordinary least squares, or Tukey reweight/refit steps |
| Iterations / Outlier limit | One to ten refits after the initial fit. Tukey cutoff is in raw units; weights are `(1-(residual/cutoff)^2)^2` inside the cutoff and zero outside |
| Minimum contrast | Source standard-deviation floor in raw units; weak texture falls back to gain one plus fitted offset |
| Gain regularization | Raw contrast scale whose square penalizes gain departure from one; zero disables this penalty |
| Minimum / Maximum gain | Inclusive usable-model bounds; out-of-range fits contribute no inspection coverage |
| Minimum fit pixels | Minimum effective sum of fitting weights, not just the number of nonzero weights |
| Minimum fit fraction | Effective support / actual in-image local window area, or / initial selected global fit count |
| Saturation | By default, exclude integer endpoint samples from both fitting and inspection. Disable explicitly if endpoints are meaningful. Floating data has no assumed saturation level |

Local moments use unnormalized [OpenCV box filters](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)
with constant-zero weight outside the image. Missing/removed neighbors provide
no evidence; the image boundary is not reflected into the fit. Each channel is
centered/scaled before float64 weighted moments to avoid squaring large raw
sample values. Robust weights are refitted independently by channel. A small
float64 variance floor also guards numerically flat windows.

Local models require nearby fit samples. A small isolated fit region cannot
support an entire large inspection area; use Global when a single photometric
model should extend from remote stable pixels. Pick a local window larger than
the defect scale you need to retain, and validate with known defects. Discontinuous
lighting, misregistration, texture changes and large defective regions can leave
residuals or bias the fitted field. Linear intensity data is preferable when a
gain/offset interpretation matters; gamma-encoded data gives empirical appearance
agreement rather than radiometric calibration.

## Outputs and judgments

- **Image** retains the original candidate. Drawable contours and a status label
  are attached as result-graphics objects in its coordinates; graphics-off skips
  their generation. At most 256 contour outlines are drawn; numeric defect counts
  include every detected pixel.
- **Adjusted** writes corrected samples only at valid inspection pixels and
  preserves original samples elsewhere, including alpha. Integer results round
  and clip to their native range.
- **Residual** is the maximum absolute compensated difference across intensity
  channels, in float64 raw units. It is zero outside valid inspection pixels.
- **Raw residual** is the maximum unadjusted channel difference, zero outside the
  selected inspection area, including values where compensation was unsupported.
- **Valid mask** must accompany adjusted residuals. An unsupported zero residual
  is not proof of agreement. **Defect mask** marks valid pixels exceeding the
  configured residual threshold in any channel.
- **Gain / Offset** are fitted float64 channel fields; unsupported models use
  neutral gain one/offset zero. They are hidden as connectors initially.
- **Fit inliers** marks requested, nonclipped fitting samples with positive weights
  in every intensity channel during the final fit. Positive weight does not prove
  a sample is defect-free or that all nearby output models have enough support.
- **Info** reports selected/fit/valid/defect counts, effective coverage, offset-only
  fallback count, settings and reference provenance. Aggregate and named-channel
  MAE/RMSE/maximum statistics compare raw and adjusted differences on the **same
  valid inspection population**. Aggregate statistics use all intensity samples;
  scalar residual images use the maximum channel at each pixel. Gain/offset ranges
  also describe that valid population.

Residuals and statistics are computed **before rounding/clipping Adjusted**. This
prevents output saturation from hiding a numerical discrepancy. A pixel fails
when its adjusted residual exceeds **Defect threshold**. Inclusive gain and residual
limits allow a float64 roundoff margin of 64 machine epsilons, scaled to gain
magnitude (at least one) or local predicted/reference sample amplitude respectively.
The maps/statistics retain their actual calculated values.

The ticket fails if no valid pixels exist, valid coverage is below **Minimum valid
fraction**, or the valid-pixel defect fraction exceeds **Maximum defect fraction**.
Reasons include `exposure_insufficient_support` and `exposure_defects`, with counts
and limits. Failed inspections still emit diagnostic data. Invalid configuration,
image types and unexpected native exceptions use the host's ordinary contained
failure handling and emit no processing result.

Each image is limited to 4,194,304 pixels, 16,384 per axis, 8,388,608 total channel
samples and finite absolute sample values no greater than 1e12. The limits bound
full-frame moment/map work. Processing is deterministic and has no device or
training state. It supports last-input preview and independent branch execution.

Open [the four original examples](<../examples/opencv tools/differential_exposure/README.md>)
for smooth lighting, an independently specified 20-pixel defect, exact 16-bit
color changes and inadequate local fitting coverage. The accompanying tests
compare local models with independent least-squares solutions and verify formats,
selections, metadata, limits, failures, XML and offscreen controls.
