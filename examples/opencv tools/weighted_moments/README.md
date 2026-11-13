# Intensity-weighted center and orientation

Open `project.json` and run once. No plugin or optional dependency is needed.
The original 16-bit image contains a uniform rectangle and an off-center bright
patch. A region selects the part. Two moment blocks compare intensity weighting
with equally weighted foreground pixels, and a final inspection region follows
the intensity-weighted pose.

The fixture generator calculates expectations analytically as a mixture of two
discrete uniform rectangles, independently of OpenCV's moment calculation:

| Measurement | Geometric foreground | Intensity-weighted |
| --- | ---: | ---: |
| Center x | 127.5 | 139.5 |
| Center y | 95.5 | 102.0 |
| Principal axis, OpenCV CCW degrees | 0 | 170.389 |
| Mass | 15,360 pixels | 122,880,000 intensity units |

The weighted center moves 12 pixels right and 6.5 pixels down. The axis is
unsigned: 170.389 degrees is the same axis as -9.611 degrees. In the weighted
block, green marks show its chosen center/axis and covariance ellipse; blue
marks show the geometric reference. Ellipses depict covariance at the configured
sigma, rather than a measured part outline.

From the repository root:

```powershell
opencv-dataflow-run "examples/opencv tools/weighted_moments/project.json"
opencv-dataflow-run "examples/opencv tools/weighted_moments/project.json" --profile lean
```

Regenerate the original image, analytic truth and graph with:

```powershell
python "examples/opencv tools/weighted_moments/generate.py"
```

## Weighting and selection

**Intensity-weighted center and orientation** wraps
[OpenCV moments](https://docs.opencv.org/5.0/main_modules/classcv_1_1Moments.html)
with explicit image, selection and coordinate handling. It measures one combined
selected support; collection tools provide separate measurements for individual
parts.

- **Intensity:** use positive contrast as each pixel's weight.
- **Uniform foreground:** use the same qualifying support, with one unit per pixel.
- **Uniform selection:** use every selected pixel, independent of intensity settings.

Contrast is `value-background` for bright polarity, `background-value` for dark,
or the absolute difference. Negative contrast is clipped to zero. Pixels must
exceed `minimum_signal` strictly; surviving weights retain their contrast value.
Masks and circle/rectangle/polygon regions intersect before support is measured.

Choose floating luminance, which ignores alpha, or a stored color channel.
Luminance preserves float64 precision for grayscale and RGB(A) inputs. Native
units retain the sample scale; normalized units map unsigned integer full scale
to [0,1], while floats retain their supplied units. Background and signal/mass
limits use the chosen units. This example uses a baseline of 1,000 native units.

## Results and validity

The default connectors are `image`, `center`, `angle_degrees`, `pose` and
`measurements`. Enable additional connectors in the inspector when needed.

| Output | Meaning |
| --- | --- |
| `image` | Original pixels, encoding and depth, preserved. |
| `center` | Valid [x,y] center in current input pixel coordinates. |
| `x`, `y` | Separate scalar coordinates when the center is valid. |
| `angle_degrees` | Reliable unsigned principal axis in OpenCV counterclockwise degrees [0,180). |
| `pose` | Existing image_pose_v1 type, usable by pose-relative regions and pose offsets. |
| `mass` | Sum of contrast weights, or qualifying pixel count in uniform modes. |
| `measurements` | Validity, support counts, covariance, variances, sigmas, anisotropy, geometric reference and center offset. |
| `support` | Uint8 mask of qualifying pixels. |
| `moments` | OpenCV-style raw, central and normalized moment dictionary with a declared working-weight scale. |
| `weights` | Optional peak-normalized float64 weight image, enabled by `emit_weights`. |
| `overlay` | Optional original-image preview with drawable centers, axes and covariance ellipses. |

Center validity requires sufficient support and mass strictly above the configured
minimum. Missing or below-limit centers fail with reasons by default. Report mode
emits an informational result with no center/pose events.

Orientation has its own variance and anisotropy limits. A circular spot or one
pixel can have a valid center while its axis is undefined. `require_orientation`
can fail such a result. Principal-axis poses require a reliable axis. Translation
only explicitly defines zero pose rotation while retaining an undefined measured
angle. Principal axes have 180-degree ambiguity; asymmetric local offsets need an
additional directional cue.

## Numerical and coordinate contracts

Weights are peak-normalized before accumulation, and the support is cropped to
keep central moments stable near large image coordinates. Reported centers and
raw moments are translated back to the input pixel grid. Reported mass restores
the selected weight units. The `moments_basis` payload records `weight_scale`:
raw/central moment values refer to peak-normalized weights and can be multiplied
by this scale for original-weight units. Normalized central moments retain the
working-weight basis. The weight image uses that same declared scale.

Tickets, frames, pyramid and companion payloads remain on the output wires. Poses
use the same frame-aware interface as other image features. Measurements refer
to the selected input grid; root-image or physical-space conversion is explicit.

Images support uint8, uint16 and float32/64, with limits of 4,194,304 pixels and
16,384 per axis. Nonfinite inputs, unrepresentable total mass, invalid selections
and native exceptions produce contained errors. Empty support and ambiguous axes
are measurement outcomes. Graphics disabled skips drawing and the optional
preview; numerical results and independently enabled weights remain available.

## Attribution

The image, project and generator are original repository examples distributed
under the repository's MIT license. No customer or third-party images are used.
