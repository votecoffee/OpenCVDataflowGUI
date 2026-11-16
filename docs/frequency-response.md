# Relative image frequency response

**Image frequency response** (`inspect.frequency_response`) compares an aligned
sample capture with a reference containing known scene detail. It measures
per-frequency linear gain and wrapped phase, suppressing bins the reference
does not excite. It does not fit registration or recover optical MTF from an
arbitrary pair of photographs.

## Inputs and calculation

Use equal-size gray/RGB/BGR(A) intensity images: uint8, uint16, float32 or float64.
Depths and encodings may differ if the selected channel exists in both. Gray
uses Rec.601 weights, `0.299 R + 0.587 G + 0.114 B`; named R/G/B/A selection uses
the semantic channel order. HSV/Lab, physical depth and premultiplied-alpha
images require explicit conversion upstream. The original sample is retained on
Image with unchanged type, channel encoding and source coordinates.

Optional sample/reference masks and regions are validated in their own image
frames. Their intersection is a shared binary selection. The None/Hann/square-root
Hann window multiplies it to produce Aperture. Empty selections/effective windows
are invalid. Both signals use exactly this aperture; neither image is cropped or
resampled internally. Source regions can be circles, rectangles or polygons.

Raw scaling retains numeric sample units. Unit range divides each unsigned image
by its own depth maximum; floats retain their supplied units. For aperture `w`,
the block optionally subtracts each signal's weighted mean, then computes:

```text
R = DFT(w * (reference - weighted_mean(reference))) / sum(w)
S = DFT(w * (sample    - weighted_mean(sample)))    / sum(w)
H = S / R
gain = abs(H)
phase = atan2(imag(H), real(H))
```

Disabling Subtract mean omits centering. The transform uses exact input dimensions
and float64, with one shared amplitude scale to prevent overflow without changing
the ratio. Coherent aperture normalization makes Absolute floor use selected
sample units, rather than image-size-dependent raw DFT sums. Supported bins exceed
all of: Absolute floor, Relative floor times the **full reference spectrum** peak,
and a numerical floor of `64 * float64 epsilon * weighted mean(abs(reference))`.
The numerical floor is evaluated before centering so cancellation roundoff in a
constant reference cannot create fictitious support.

Accepted complex denominators are normalized to unit magnitude before native
`divSpectrums`, avoiding its epsilon bias at weak bins. See OpenCV's
[DFT and spectral division reference](https://docs.opencv.org/5.0/main_modules/core_array.html).
This is a direct single-pair ratio, without regularized deconvolution, multi-pair
averaging or a statistical coherence/confidence estimate.

## Frequency selection and results

Minimum/Maximum frequency select an inclusive radial band in cycles per input
pixel. The default maximum 0.5 is a disk; diagonal corner bins can extend to
`sqrt(0.5)`. Direction selects All, Horizontal, Vertical or an Angle cone plus
its antipodal counterpart. Horizontal frequency varies across image x; angle
uses x right/y down. Angular tolerance is an inclusive half-angle. DC is excluded
unless Include DC is enabled and the minimum frequency is zero; centering usually
removes its energy anyway.

Gain, Phase, Supported and Phase valid are full-size maps with DC centered using
`fftshift`, including odd dimensions. Metrics supplies signed Frequency x/y vectors
for those map indices. Unsupported gain values are zero placeholders. Phase also
requires sample amplitude above its relative/numerical floor; undefined phase is
zero with Phase valid clear. A zero sample can have valid zero gain and no defined
phase. Always consume the corresponding masks when interpreting values.

Equal-width radial rows report supported/phase-valid counts, mean gain and circular
mean phase. Empty bins are null. Circular phase is also null when opposing angles
cancel. Conjugate partners are both counted; counts do not represent independent
measurements. Radial averaging can hide directional differences; use the cone or
full maps for anisotropic behavior. The diagnostic plot omits unsupported rows.

Pass requires the configured minimum supported count/fraction and inclusive
minimum/maximum mean gain. Fraction divides supported bins by eligible band/cone
bins. Mean gain is unweighted across supported bins. Check phase additionally
requires defined phase and limits its maximum absolute wrapped distance from zero.
Mean gain comparisons tolerate 128 machine epsilons relative to each boundary;
phase limits tolerate 128 epsilons times 180 degrees for numerical roundoff.
Reasons are `frequency_reference_support_insufficient`,
`frequency_gain_out_of_tolerance` and `frequency_phase_out_of_tolerance`.

With no excited reference, the block emits failed diagnostics/maps and no Mean gain
or Maximum phase scalar events. Unknown metrics remain null, rather than passing
or reporting a fabricated zero response. Invalid settings/frames/resources reject
all outputs. Unexpected native failures use normal exception containment/logging.

## Provenance, alignment and limits

Pixel grid (default) is the caller's assertion that equal pixel indices represent
aligned scene points. Same frame requires matching known root ID, root/image sizes
and image-to-root matrices within absolute 1e-9. Neither proves physical alignment;
a real residual shift appears as a phase ramp. Independent tickets allow a saved
historical reference. Same ticket requires matching acquisition identity.

Image/Aperture retain the sample's spatial/pyramid context. Frequency maps have a
separate `:frequency` root in bin-index space, a frequency-grid descriptor and the
original spatial/pyramid record under `spectral_source_grid`. Metrics retains both
source contexts/tickets. Frequencies describe the current input pixel pitch,
including when that input is a smaller pyramid level; they are not automatically
converted to cycles per millimeter. Result text belongs to the original sample
frame. Graphics-off/Lean keeps primary maps, radial measurements and judgments.

The preflight allows 2.1M pixels, sides up to 16,384 and a configurable 1–256 MiB
work-buffer estimate before channel extraction/DFT allocation. The estimate is
256 bytes per pixel plus radial-row overhead. Runtime snapshots, graph/history
copies, previews and native allocator overhead are additional. Crop upstream when
the dense analysis exceeds limits. Inputs and native state are not shared or
modified; independent blocks can run in parallel.

Windowing, masking, boundary differences, noise, gamma and scene changes can bias
the measured ratio. A filtered finite crop generally differs from filtering a
windowed/selected signal. The [ten original examples](<../examples/opencv tools/frequency_response/README.md>)
use periodic binomial filtering, known integer shifts and exact brightness changes
for independent references. They are numerical inspection fixtures, not optical
MTF certification or a general camera-calibration measurement.
