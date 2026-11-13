# Rotation and scale registration with log-polar proposals

Open `project.json` and run once. An original 16-bit RGB part was rotated by
145 degrees, scaled by 0.93 and translated by `[-3, 6]` pixels. **Register
rotation and scale (log-polar)** must recover the inverse moving-to-reference
transform, resolving its 180-degree spectral alternative through spatial checks.

On the pinned build the recovered angle is approximately `-145.002°`, scale
`1.07512` and normalized correlation `0.9983`. Supported-pixel image RMSE drops
from about 13,432 to 798 counts. Resampling prevents an exact pixel round trip;
the independent control-point comparison must stay within 0.3 original pixels.
`assets/truth.json` records the directly calculated sine/cosine matrices and
control points, without using the registration algorithm.

| Project | Expected result |
| --- | --- |
| `project.json` | Pass; recover the inverse rotation, scale and translation. |
| `symmetric.json` | Deliberate failure: equally plausible 180-degree poses withhold the primary image and transform. |
| `orientation_prior.json` | Pass: the same symmetric image with an explicit `[-45°,45°]` prior resolves the permitted pose. |

## Proposal and verification stages

`inspect.logpolar_registration` takes a moving **Image** and equal-size
**Reference**, supporting gray/RGB(A)/BGR(A) uint8/uint16/float32/float64.
The reference defines the aligned pixel grid; the moving image owns the active
ticket. Both source grids and identities remain in the payload. Straight alpha
is resampled with the image and contributes captured support; premultiplied
images must be unassociated first.

Each image is scaled safely before luminance conversion. A square working canvas
uses isotropic sampling, including for rectangular or reduced inputs. Centered
DFT magnitude is remapped into semilog radius and angle. Native phase correlation
provides a rotation/scale seed; additional wrapped correlation peaks and every
180-degree alternative provide bounded hypotheses. Configurable inner/outer
frequency bands, magnitude compression, spatial/radial Hann windows, angular
centering and polar sampling control this stage.

Each hypothesis is rotated/scaled and translated using spatial phase correlation.
Optional affine ECC refinement is projected back to an orientation-preserving
similarity. Excessive anisotropy, out-of-range motion and inadequate captured
support reject the proposal. Final scoring uses normalized luminance correlation
with linear sampling, independently of the final image interpolation setting.

The best pose must meet **Minimum score**. Distinct hypotheses must have a
sufficient best-minus-runner-up **Minimum margin**. **Pose separation** merges
near-duplicate estimates using RMS displacement of their mapped source corners
in original pixels. Search bounds and the finite peak budget define the tested
alternatives; uniqueness is conditional on that search. Zero margin explicitly
disables the margin gate. Correlation and phase response are not probabilities.

## Selection and coordinate contracts

Optional **Mask** and **Region** select spatial verification pixels in the
reference grid. They do not crop the global Fourier proposal. Use upstream
cropping/masking to remove unrelated clutter from the seed. **Moving mask**
defines captured support in the moving grid. Alpha-zero samples are unsupported.
Minimum overlap, pixel count and luminance variation prevent tiny or flat
comparisons from producing a pose.

**Transform** maps moving/source pixels to reference/target pixels.
**Geometry** supplies its inverse `output_to_input` map. Both describe original
pixel coordinates, even when the working canvas is smaller. Their explicit
source/target contexts allow **Transform image points** to preserve the correct
frame in either direction. Aligned image and map outputs retain the reference's
coordinate and pyramid context, moving ticket and companion data.

The accepted **Image** preserves moving dtype and channel order. Final
interpolation and border filling are configurable. **Valid mask** combines the
reference selection with conservatively captured source support, excluding
extrapolated pixels and interpolation margins. The image still covers the whole
reference canvas, so use the mask for downstream scoring.

Failed or ambiguous matches emit no primary aligned image, transform, geometry,
angle or scale. **Info**, **Candidates**, original Source and Reference images
remain available for diagnosis. Candidate matrices are diagnostic evidence, not
accepted poses. Drawable axes/labels use the reference grid, and disabling
graphics skips their construction.

## Controls and limits

Angles are moving-to-reference OpenCV degrees in `[-180,180)`. Scale is also
moving-to-reference; a source enlarged by 1.25 normally needs alignment scale
0.8. Angle bounds describe one non-wrapping interval. Translation bounds limit
center displacement in original pixels; zero chooses 40% of the image size.

ECC iteration count zero requests coarse-only verification. By default failed
refinement rejects that proposal. **Coarse** fallback explicitly permits the
unrefined proposal to face the same spatial checks. Images require axes of
32–16384 pixels and at most 4,194,304 pixels. Working size is capped at 1024;
the hypothesis budget is twice `1 + Polar peaks`. Native iterations are bounded.

This is whole-image similarity registration. It does not solve perspective,
occlusion-aware multi-part search, or arbitrary repeated-pattern ambiguity.
Severe clipping, unrelated clutter, repetitive translation peaks or a poor
Fourier seed can prevent recovery. Independent geometry/image verification is
still needed for an actual application.

## Attribution and regeneration

All part images, projects, code and truth data are original work under this
repository's MIT license. The symmetric fixture is an exact array-reversal union
of the original part, providing an independently known 180-degree ambiguity.

```powershell
.\.venv\Scripts\python.exe "examples/opencv tools/logpolar_registration/generate.py"
```

Algorithm references: [OpenCV polar mapping](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html)
and [phase correlation](https://docs.opencv.org/5.0/main_modules/imgproc_motion.html).
