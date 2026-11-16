# Resample measured image profiles

`cv.resample_image_profile` aligns path measurements to an exact common pixel or
physical pitch. It uses an original Float64, rate-aware Kaiser implementation,
and appears under **Custom > Image processing > Statistics**. It is a tested
alternative to the native signal resampler, with explicit image-coordinate and
invalid-support contracts. It does not expose general matrix or audio processing.

## Build the workflow

Connect **Path intensity profile > Profile** to **Resample image profile > Profile**.
The path block applies image masks/regions first and records excluded samples as
null. Two resamplers can share a grid: configure the first with its own/explicit
extent, connect its Profile to the second's Reference, then select Reference grid
on the second. Reference values and image coordinates never replace current data.

For physical distances, first rectify the image with **Rectify workplane**. Send
its image through the path block and its Scale to the resampler. The scale must
describe the profile's exact current grid and ticket. Pixel distances convert to
that workplane's mm/cm/m/in units; source points stay in current-image pixels.
Already physical profiles cannot be scaled again. Reference distance units must
match. This uses a declared calibration; it does not qualify that calibration.

Distance origin adds a taught offset after scale conversion. It aligns known
path origins; it does not find correspondence or infer registration. Arc distance
is relative to the sampling path, rather than automatically becoming world X/Y.
Reference profiles can come from another acquisition; their provenance is recorded
separately and does not become the current ticket/frame.

## Grids and profile data

Input is the legacy Path intensity profile result or `image_profile_v1`, with
`closed`, `path_length`, uniformly increasing `distances`, `points` and named
`channels` (`channel`, `values`). Versioned output also declares `units`. There
must be 2-8192 samples, 1-4 unique channels, finite numeric coordinates and finite
values or null. A measured value requires a source image point. Nonuniform,
missing-distance and nonfinite inputs are rejected; null values represent gaps.

Own extent starts at the first source coordinate. Explicit extent can include
unmeasured coordinates outside the source; those stay null with null source
points. Pitch stays exact: a partial final interval is omitted, never stretched
to force the last endpoint. Reference grid uses its exact distances and pitch.
Closed profiles wrap once without duplicating the first sample; the target pitch
must divide one whole path period and a closed reference must match that period.

Outputs are Profile, source-image Points, channel-by-sample Valid flags,
Selected count (samples with at least one valid channel), and Report. The new
profile can be resampled again. Source fractional indices and interpolated image
points explain where each output lies. Geometry interpolates the input sample
polyline; it cannot recover an unsampled bend in the original path.

Distances and image coordinates remain separate. Ticket, spatial, pyramid,
recipe and other payload metadata retain current-source meaning. A different
reference image grid is allowed because only its distance grid is consumed.
No affine image transform is invented for a one-dimensional plot.

## Filtering, boundaries and gaps

Kaiser sinc performs normalized, phase-aware interpolation. When downsampling
with Antialias enabled, its cutoff follows target Nyquist times Rolloff and its
support expands with the pitch ratio. Linear mode uses linear interpolation,
with a normalized Kaiser FIR prefilter when antialiased decimation is requested.
Turning antialias off permits aliases and is useful as a comparison baseline.
Kernel lobes, window beta and rolloff expose the quality/support tradeoff.

Every non-negligible contributing sample must be valid for that channel. An
invalid gap therefore widens with filter support instead of being bridged or
filled with zero. Other channels can stay valid. Identity sampling preserves
isolated gaps without unnecessarily invalidating adjacent exact samples.

Reflection and replication extend only the outer boundaries; targets outside
the measured extent stay invalid. Report includes per-output `boundary_extended`
and `edge_extension_count`, so inspection can exclude values using that boundary
assumption. Invalid boundary mode rejects unsupported kernels. Closed profiles
use periodic support. No reflection or replication fills an interior null gap.

Constants retain their value and low-frequency intensity amplitude is preserved
within the tested bounds. These are sampled intensities, not integrated counts;
resampling does not promise preservation of a sum over sample entries. Sharp
features can ring under sinc interpolation. Do not interpret an interpolated peak
as independently measured subpixel metrology.

Per-channel diagnostic plots break at invalid runs and label distance units and
origin. Drawable sampling paths retain the Profile input's image frame for
optional final overlays. Graphics-off skips these diagnostics, not the numerical
work. An entirely unsupported result emits explicit data with a failed inspection
and `profile_empty_support`; malformed input/work limits use normal invalid-input
handling with no output event.

The ratio is bounded to 256 in either direction, coordinate magnitudes to 1e9,
and target count to 8192. A resolved local origin is required for numerical pitch
precision. Maximum work bounds output count times kernel taps times channels.
Source image points are bounded to 1e7 and input magnitudes to 1e100. These are
allocation/work guards, not a promised latency or native-memory measurement.

## Native signal qualification

OpenCV documents cubic interpolation and Kaiser/FIR filtering in
[resampleSignal](https://docs.opencv.org/5.0/extra_modules/signal.html).
The [5.x implementation](https://github.com/opencv/opencv_contrib/blob/5.x/modules/signal/src/signal_resample.cpp)
has its own endpoint convention and conditional filtering. On the supported
OpenCV 5.0.0 contrib 5.0.0.93 wheel, a 1024-sample 0.30-cycle/input-sample sine
reduced 4:1 retained interior RMS 0.6298; the target Nyquist is 0.125. A ramp's
endpoints were 0.4375 and 1023.0625, rather than the exact 0..1020 distance grid
used here. Upsampling a constant also changed the edge values. Native frequency
arguments must be at least 1000, which is not a physical pitch API.

Those observations do not qualify the native function for this workflow.
The supported block uses independently tested formulas rather than copying the
native implementation. It does not claim native `signal.resampleSignal` coverage.

## Reproducible examples

[Three original projects](<../examples/custom tools/profile_resampling/README.md>)
use a 16-bit image with known low/high-frequency terms and a Gaussian peak.
They compare grids from two source sampling densities, exercise a declared
workplane scale, and compare filtered/unfiltered masked decimation. Unit checks
use analytic signals for amplitude, phase/peak, alias suppression, boundaries,
closed wrapping and channel-specific support, plus ticket/frame, resource,
XML and editor controls. These qualify the image-profile baseline, not a
measurement-system calibration or production acceptance.
