# Signal/background contrast and visibility

**Custom > Inspection > Visibility > Measure signal/background contrast** measures
two selected portions of the same image. It reports means, standard deviations,
signed differences, CNR and mean-based normalized contrasts. Optional limits
test one selected channel. It preserves the original image, encoding, ticket,
acquisition start time, spatial/pyramid context and companion payloads.

Try the [original strong/weak example](<../examples/custom tools/region_contrast/README.md>).
Its two 16-bit RGB captures deliberately produce one pass and one failure.

## Selections

Connect a signal mask, signal region, or both. When both are supplied, their
intersection is used. Explicit background mode requires a background selector
with the same semantics. Masks must be uint8 and match the image grid. Regions
can be created with the existing rectangle, circle, polygon or pose-relative
region tools. Known coordinate-frame mismatches fail before measurement.

The optional common mask/region defines valid working pixels for both regions.
Outside-signal background mode uses the remainder of that work selection;
disconnect explicit background selectors when choosing it. Missing or empty
selectors never silently mean the whole image. Each final region must meet
**Minimum pixels**; sample standard deviation also requires at least two.

Overlap defaults to rejection. **Exclude signal** removes shared pixels from
the background before statistics; **Allow** deliberately shares samples.
The report contains the original overlap count and final region sizes.
These are descriptive spatial statistics, not a test of independent samples.

## Definitions

Let `S` and `B` be the selected signal/background means and `s` and `b` their
standard deviations in stored image-sample units.

| Report field | Definition |
| --- | --- |
| Difference | `S-B`, retaining the sign |
| Absolute difference | `abs(S-B)` |
| Measured noise, Quadrature | `hypot(s,b)` |
| Measured noise, Background | `b` |
| Effective noise | `max(measured_noise, noise_floor)` |
| CNR | `abs(S-B)/effective_noise` |
| Weber contrast | `(S-B)/B` |
| Mean-based Michelson contrast | `(S-B)/(S+B)` |

Population deviation divides the centered squared deviations by `N`; sample
deviation uses `N-1`. OpenCV's native population reduction is applied to
centered/scaled samples to preserve low contrast on a large DC level. The
means and deviations are returned in original sample units. No per-image
brightness normalization or conversion to uint8 is applied.

The quadrature CNR convention is also used in this
[primary contrast-to-noise study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10481948/).
The alternative background-only denominator is explicitly selectable because
different applications use different noise definitions. Within-region texture,
gradients and actual defects all contribute to the measured spatial deviation;
this is not a separate sensor-noise estimate or detection probability.

Weber and Michelson require nonnegative selected samples and positive respective
denominators. They compare representative region means, rather than global
image extrema. Signed images still support differences and CNR. Zero measured
noise leaves CNR undefined unless a positive raw-unit floor is explicitly set.
The block never invents an infinite visibility score for constant patches.

## Channels, limits and undefined values

Gray mode uses floating BT.601 weights on decoded BGR/RGB samples, avoiding
integer grayscale rounding. Native mode reports each stored channel using its
actual encoding labels. Alpha is excluded by default; including it adds a
numeric channel and does not weight or hide transparent pixels. Mask those
pixels explicitly if appropriate. HSV hue and other native channels use ordinary
linear arithmetic, not circular statistics or perceptual color distance.

**Channel** chooses the scalar outputs and decision channel. Auto uses the first
reported channel. **Value** is the magnitude of the chosen metric; signed
contrasts remain in their individual ports and the report. Threshold checking
is off by default. When enabled, Value must meet **Minimum value**, and Bright
or Dark requires a positive or negative mean difference. Equality satisfies
only Either. Other reported channels do not silently participate in the decision.

Missing, empty, insufficient or rejected overlapping selections produce a failed
diagnostic result with real masks/counts but no measurement scalars. A valid
selection can still have an undefined selected metric. Its **Value** and that
metric's port emit no event; independently defined means/differences can still
be consumed. **Measurements** identifies undefined fields and reasons.

Overflowing/underflowing physical-unit outputs are withheld. A dimensionless CNR
can remain valid even when a raw difference or sample deviation cannot fit in
float64. A threshold failure keeps its real finite measurement and assigns
`contrast_below_minimum` or `contrast_wrong_polarity`. Undefined/missing evidence
has a distinct reason. These are tool outcomes; a final ticket judgment remains
the responsibility of a final-judgment block. Optional scores are bounded
threshold fulfillment, not confidence probabilities.

## Results and execution

The node initially exposes Image, Value, Valid and Measurements. Expose individual
means, deviations, contrasts, masks and counts in the sidebar as needed. Image
pixels stay unchanged. Drawable result objects outline signal in cyan and
background in orange and label the selected metric/status. Graphics-off avoids
contour creation. Up to 64 contours per selection are drawn; overly long contours
are simplified for display. The report flags limited outlines, while measurements
and mask outputs always use the full exact selections.

The block supports uint8, uint16, int16, float32 and float64 images, up to
16,777,216 pixels. Gray conversion of HSV/Lab follows native supported color
depths; native-channel measurement avoids that conversion. The normal block
framework provides exception isolation, input failure policies, logging, timing,
parallel branch scheduling and JSON/XML configuration persistence.

Native reference: [OpenCV mean and standard deviation](https://docs.opencv.org/5.0/main_modules/core_array.html).
