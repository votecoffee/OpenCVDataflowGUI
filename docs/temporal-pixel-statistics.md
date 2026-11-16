# Temporal pixel statistics

`sequence.pixel_moments` (**OpenCV → Image Processing (imgproc) → Temporal
statistics**) measures stationary temporal variation at each pixel. It wraps
native accumulation functions using centered float64 moments. Choose one named
channel for X and Y: gray, blue, green, red or alpha. Gray uses linear BGR luma
coefficients without integer rounding. RGB/BGR and alpha tags are respected;
convert HSV/Lab to gray/RGB before using this tool.

Y comes from **Other** when connected, otherwise from Image. Selecting the same
channel twice gives self-covariance equal to variance. Use multiple blocks or
channel splitting to measure additional channel pairs. Input values keep their
numeric units; there is no automatic uint8/uint16/float normalization. All pixels
must be finite, with absolute values ≤1e100. Physical `DepthArray` values require
an explicit conversion to an ordinary measurement image.

## Numerical outputs and validity

Expose the desired connectors in the inspector. The block provides float64
`mean_x`, `mean_y`, `variance_x`, `variance_y`, `std_x`, `std_y`, `covariance`,
`correlation`, `count` and `effective_count` maps, plus uint8 `valid` and
`correlation_valid`. Counts are exactly represented integers up to Maximum
samples. Variance is in source units squared, covariance in X units × Y units,
standard deviation/mean in source units, and correlation is unitless.

Only pixels within the current mask/rectangle/circle/polygon selection contribute
observations. Counts and weight histories are per pixel. Excluded pixels are
not inserted as zero-valued observations. A newly included pixel starts its own
history. Numerical maps are zero outside valid current support; **Valid**, not
a zero numerical value, determines whether a measurement exists. Count/effective
count retain available support inside the selection even below the validity
threshold. Correlation also requires both variances to be positive; its separate
validity mask distinguishes undefined correlation from a genuine zero value.

Readiness requires Minimum samples per pixel, Minimum coverage of the current
selection, and the shared Warmup activation threshold. Before readiness there
are status events but no primary map events. An empty selection is not ready.
**Learn=false** freezes existing state; sufficiently supported selected pixels
can still emit results. Each pixel freezes at Maximum samples. Status records
updated pixels, count bounds, effective support, RMS X deviation, alignment
evidence and last learning tickets, separately from the current input ticket.

## Cumulative and exponential observations

**Cumulative** gives every accepted observation weight 1. **Exponential** gives
eligible observations weights `alpha * (1-alpha)^age`, with age counted per
pixel observation. Pausing or masking a pixel does not decay it. These are not
wall-clock/time-interval weights. The weighted mean is normalized from the first
observation rather than starting with artificial zero-image observations.

Let `W=sum(weights)` and `Q=sum(weights²)`. Population normalization divides
centered moments by W; sample normalization divides by `W-Q/W` (N−1 for equal
weights). Effective count is `W²/Q`, describing weight concentration. Alpha=1
is allowed for population statistics, but has no sample-variance degrees of
freedom and is rejected with sample normalization.

The first observed value is a per-pixel reference offset. Centered online means,
square sums and cross products avoid subtracting large uncentered squared sums.
Tests compare independent batch calculations and a Decimal reference for small
variations around 1e12; ±1e100 samples also remain finite. Floating-point input
quantization, underflow and roundoff still limit measurable noise. Correlation
is clipped to [-1,1] for roundoff. Observation counts do not prove independence,
and sample normalization does not remove temporal autocorrelation.

## Paired sources and metadata

A connected Other must emit and have the same dimensions. **Same frame** checks
declared spatial grids; missing grid metadata permits dimensional comparison
only and is reported as such. **Pixel indices** explicitly asserts that the two
grids can be compared by index. It does not register the images. Ticket policy
can match when both tickets exist, require both matching tickets, or explicitly
ignore them. Matching ticket labels/metadata do not prove exposure synchronization.

Maps use Image's current ticket, spatial/pyramid context and arbitrary payloads.
The paired source's shape, depth, encoding, grid, recipe and source identity are
part of the temporal context. Recipe/grid/settings/encoding changes, reset,
stream limits and ingress-gap behavior use [sequence memory](sequence-memory.md).
All parameter changes, including preview settings, follow that shared context
reset/separation policy. Use Learn input for pause/resume without reconfiguration.
Each eligible activation is an observation; no historical-ticket deduplication
is inferred. Saved JSON/XML holds configuration, not learned numerical state.

## Resources and viewing

The model holds ten float64 planes per stream, with a preflight state estimate
and a pixel/workspace guard. The workspace estimate covers working maps and
input sample buffers; native internals and runtime/editor snapshots can require
more memory. These are not hard RSS or native execution-time limits. Work is
serial per block, state is copyable across editor workers, and native exceptions
clear the affected stream without publishing partial maps. Preview execution is
disabled because it would count an additional observation; use normal Run once.

Debug results include a diagnostic heatmap for the selected metric and preview
range. Correlation always uses [-1,1]; unsupported pixels are black. The raster
is a diagnostic map, separate from drawable result geometry. Graphics off/Lean
skips it while retaining numerical outputs. A range change does not rescale the
numerical maps, although reconfiguration can reset the model as described above.

Use stationary/registered scenes and masks to exclude expected moving parts.
Observed variation also contains scene movement, lighting drift and process
changes; this tool does not certify camera noise or physical registration.
The [seven original replay examples](<../examples/opencv tools/temporal_pixel_statistics/README.md>)
include known 2/4-unit sample deviations, channel/source covariance, exponential
weights, region support and deliberate ticket/memory failures.

Native APIs: [OpenCV accumulation](https://docs.opencv.org/5.0/main_modules/imgproc_motion.html).
Weighted normalization: [NumPy covariance notes](https://numpy.org/doc/stable/reference/generated/numpy.cov.html).
