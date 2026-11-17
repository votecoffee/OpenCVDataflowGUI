# Estimate a uniform linear-motion PSF

`inspect.estimate_motion_blur` proposes a centered line point-spread function
(PSF) from one degraded image. It searches a bounded length/angle grid on
approximately isotropic texture and validates the chosen hypothesis on separate
image tiles. Accepted PSFs connect directly to `cv.wiener_deblur`.
The [eleven projects](<../examples/custom tools/motion_blur_estimation/README.md>)
show known motion, restoration, noise sensitivity and rejected hypotheses.

This is a Custom inspection algorithm built with native OpenCV DFT operations.
It is not an OpenCV blind-deconvolution binding or an unrestricted blur solver.
No additional packages, neural models or training are required.

## Image and selection

Supply a nonempty gray, RGB, BGR, RGBA or BGRA image at uint8, uint16, float32 or
float64 depth. Luminance is the default; R, G or B can be selected by name.
Original pixels, encoding, ticket, processing start time and spatial/pyramid
context remain unchanged on Image. Estimates use **current-image pixels**, not
an implicitly rescaled largest pyramid level. The PSF's metadata records its
source spatial/pyramid context without pretending the kernel table is an image
in the source coordinate frame.

Mask and Region intersect on the source grid. Only entire selected square tiles
contribute; there is no black padding around partial selections in the spectrum.
Positive straight alpha is required throughout each tile. Premultiplied input
requires unassociation. A selection with too few complete tiles yields a normal
failed estimate and no PSF. The source image remains available.

Tiles are placed on a grid anchored at the image origin, with no overlap. If
there are more than Maximum tiles, evenly spaced entries in row order are used.
Alternating selected tiles form the fit and validation groups. Minimum tiles is
at least four, so both groups have at least two. The partition and tile origins
are reported. These groups are disjoint observations, but neighboring texture
and blur neighborhoods can still be statistically correlated.

## Spectral model and search controls

Each tile is mean-centered and multiplied by a fixed periodic separable Hann
window. Native DFT powers are averaged separately for fit and validation. The
selected annulus excludes DC, frequencies above the configured limit and the
redundant conjugate half. At least 64 nonredundant bins over three radial rings
are required. Log power is centered within radial rings to remove an unknown
isotropic spectral envelope.

Each candidate is the Fourier power of a centered, bilinearly integrated uniform
line PSF. Its expected Hann-window power is smoothed with the separable
`[1,4,1]/6` frequency kernel. A searched positive constant spectral floor models
unexplained isotropic power. **Model floor is a nuisance parameter, not a measured
noise-to-signal ratio or a replacement for Wiener regularization.** The model
and radial whitening are approximations when texture is colored or nonstationary.

Length runs from Minimum length in Length step increments without exceeding
Maximum length. It is bounded by one quarter of Tile size and 63 pixels.
Angle start/span/step define an arc; 180° covers all undirected orientations.
Angles are clockwise from horizontal in image coordinates and wrap modulo 180°.
Motion sign and the kernel's positional phase are not inferred; the emitted
kernel imposes a centered origin.

The fit score is explained radial-whitened log-power variance, rather than a
probability of correct motion. Fit data alone selects length, angle and model
floor. Validation then scores that one hypothesis. Validation never selects a
different candidate, refits the parameters or triggers a retry.

Minimum score must be met in both groups. Minimum margin requires separation
from the highest-scoring distinct alternative on fit data. Distinct length and
Distinct angle define that separation; changes of model floor alone do not make
a new geometric hypothesis. A zero margin explicitly disables ambiguity gating.
With a positive margin, a search containing no distinct alternative is rejected.
Boundary policy normally rejects a winning length endpoint or partial-angle-arc
endpoint, since the true solution might lie beyond the search. A full 180° arc
has no angular seam boundary. Allow explicitly accepts endpoint estimates.

Minimum contrast is mean tile standard deviation in common working units:
unsigned full scale, or the maximum absolute selected color value for float
images, with one for an all-zero image. Normalization precedes luminance and DFT
work, supporting tiny and HDR values. Alpha does not set the color scale.

## Acceptance, ports and restoration

| Output | Contract |
| --- | --- |
| Image | Original source pixels and source acquisition context. |
| PSF | Accepted odd-sized, centered, nonnegative, unit-sum kernel table. |
| Length / Angle | Accepted current-pixel line extent and modulo-180 orientation. |
| Report | Assumption, selection/partition, contrast, best/alternative hypotheses, scores, margin, boundary, work and rejection reason. |
| Diagnostic | Optional fit/validation/candidate frequency plot on its own coordinate grid. |

Insufficient tiles/contrast/structure, weak fit, failed validation, ambiguity and
rejected search endpoints are normal inspection failures. **PSF, Length and
Angle emit no event on rejection.** Report's candidate remains explicitly
tentative. Invalid settings, malformed images/metadata and unexpected native
exceptions use the common contained error path; later activations can recover.

Wire PSF to Wiener deblur's optional connected PSF input. That input is required
when connected: rejected estimates cannot silently use a configured default
kernel. Supply the original image to Wiener, choose its boundary model and tune
Noise to signal independently. Its own image-size/boundary/output limits still
apply. The examples use circularly blurred controls and Circular restoration;
real image boundaries require separate qualification.
A warm-graph check verifies a later rejected acquisition does not reuse a prior
accepted PSF or emit a new restored image, and a following valid acquisition
resumes restoration normally.

Drawable accepted line/annotation graphics belong to the original image grid.
Rejected cases have a rejection annotation. Diagnostic plots have independent
frequency coordinates. Global/per-block graphics disabled omits those optional
artifacts without changing estimates. The common executor provides logging,
timing, tickets, errors, persistence and last-input previews.

## Scope, bounds and evidence

Images are limited to 4,194,304 pixels and 16,384 per axis. Array work defaults to
a 512 MiB estimate checked before image preparation. Total candidate trials
include every length/angle/floor combination and default to a 4,096 cap. Search
work counts tile pixels, template pixels and trial frequency samples; its default
cap is 64 million, checked before native spectra. The estimate excludes graph
copies and native workspaces and is not a runtime or process-memory guarantee.
Templates are visited one at a time, with no persistent cache or mutable model.

Independent centered-path weights verify exact horizontal kernels and moments.
Examples use a separate trapezoidal pixel-space integration, known lengths and
angles, and a disconnected sharp reference. Qualified cases recover parameters
within one pixel/five degrees and lower restored MSE below 80% of degraded MSE
on these fixtures. Flat, sharp, heavily noisy, mixed-blur and ambiguous controls
withhold PSFs. Color/depth/alpha, tiny/HDR units, masks/regions, disjoint validation,
source timing/coordinates, budgets, error/recovery, XML and offscreen controls
have focused checks.

Anisotropic scene content can mimic blur, and matching patches can share a wrong
model. Spatially varying, curved, rotational, defocus, Gaussian, mixed and
arbitrary kernels are outside this estimator's family. Saturation, compression,
gamma processing and changing scene spectra can invalidate its assumptions.
Neither a score nor improved synthetic MSE certifies real defect recovery or
camera motion. Qualify the model and controls on representative captures.
OpenCV's [motion restoration tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/motion_deblur_filter/motion_deblur_filter.html)
documents the known-PSF Wiener path; this estimator supplies a separately tested,
conditional hypothesis rather than an OpenCV-provided unknown-kernel solver.
