# Local structure tensor and orientation

Open `project.json` and run once. It needs the standard application installation,
with no plugin or downloaded model. The original 16-bit fixture contains vertical,
diagonal and horizontal textures, a corner, a flat patch and a weak texture.

**All local orientations** shows the entire image, including labels and panel
boundaries. Six selected probes exclude those boundaries and test expectations
stored independently by the fixture generator in `assets/truth.json`:

| Probe | Expected tangent | Result on the pinned OpenCV 5 build |
| --- | --- | --- |
| Vertical texture | 90 degrees | All 13,056 selected pixels valid; zero angle error. |
| Diagonal texture | 135 degrees | All 13,056 selected pixels valid; zero angle error. |
| Horizontal texture | 0 degrees | All 13,056 selected pixels valid; maximum axial error below 0.001 degrees. |
| Corner center | Undefined | Both eigenvalues are strong, but coherence is below the limit. |
| Flat patch | Undefined | Zero gradient energy; no valid orientation. |
| Weak texture | Undefined at the default limit | Energy below the configured minimum; no valid orientation. |

The corner probe selects one pixel at the quadrant intersection. Its surrounding
straight edges have valid orientations; they are outside this probe. The last
three probes emit valid informational reports and zero validity masks, with no
`dominant_angle` event. An inspection can judge those reports using its own rules.

From the repository root:

```powershell
opencv-dataflow-run "examples/opencv tools/structure_tensor/project.json"
opencv-dataflow-run "examples/opencv tools/structure_tensor/project.json" --profile lean
```

Regenerate the original image and graph with:

```powershell
python "examples/opencv tools/structure_tensor/generate.py"
```

## Outputs and confidence

**Local structure tensor and orientation** wraps
[OpenCV's cornerEigenValsAndVecs](https://docs.opencv.org/5.0/main_modules/imgproc_feature.html#cornereigenvalsandvecs).
It sorts the eigenpairs, retains nonnegative eigenvalues and computes confidence
before reporting a direction.

| Output | Meaning |
| --- | --- |
| `image` | Original image, with its encoding, pixel depth and values preserved. |
| `orientation` | Float32 unsigned angle in degrees, in [0,180). Check `valid` before using it. |
| `coherence` | Float32 `(largest-smallest)/(largest+smallest)`, in [0,1]; zero for flat areas. |
| `valid` | Uint8 mask: 255 for a reliable orientation, zero otherwise. |
| `dominant_angle` | Energy-times-coherence weighted axial mean over valid pixels. No event when there is no direction or balanced perpendicular contributions cancel. |
| `measurements` | Selected/valid counts, valid fraction, mean energy/coherence, angle, concentration and coordinate/intensity conventions. |
| `eigenvalue_max`, `eigenvalue_min` | Sorted float32 eigenvalue maps. Ambiguous corners can have two strong values. |
| `energy` | Sum of the eigenvalues, in working-luminance squared units. |
| `vector_x`, `vector_y` | Unit vector along the chosen unsigned orientation at valid pixels; zero elsewhere. |
| `orientation_preview` | BGR visualization: angle as hue, coherence as saturation, invalid pixels black. |
| `overlay` | BGR preview with sparse colored direction marks on the original image. |

The first six outputs are exposed as connectors by default. Enable additional
connectors in the inspector when needed. The Results tab includes the numeric
maps and previews. Double-click an image to inspect it in a larger window.

`min_energy` is a strict lower limit. `min_coherence` is inclusive. Flat areas,
equal eigenvalues and unavailable eigenvectors remain invalid even with both
limits set to zero. Invalid orientation/vector values are zero, so a zero angle
alone cannot distinguish a horizontal direction from an undefined direction.

Angles use image axes: +x right and +y down. **Tangent** follows an edge;
**normal** follows its strongest gradient axis. Directions are unsigned: values
near 180 degrees share the axis near zero. Compare using axial distance
`abs((actual - expected + 90) % 180 - 90)`, rather than an ordinary angle difference.
The concentration report indicates agreement of the contributing axes.

## Parameters, selection and limits

The neighborhood size, Sobel aperture or Scharr operator, and image border mode
are configurable. Masks and rectangle/circle/polygon regions intersect to select
output centers. Neighborhoods still read the full source image; a selection
boundary does not introduce an artificial intensity edge. Numeric maps are zero
outside the selection, while the original `image` output remains complete.

Integer luminance uses its full uint8/uint16 range as [0,1], with no normalization
based on the current image's extrema. Float32/64 images retain their supplied
intensity units in a float32 working copy. Changing intensity scale, aperture or
neighborhood changes energy; set the threshold for that configuration. Oversized
images above 2.1 MP and nonfinite working tensors produce contained errors.

All maps retain ticket, frame, pyramid and companion payloads. Their numeric
angles/vectors refer to the current input pixel grid; they are not automatically
converted to the root image or a physical frame. Transform directions using the
appropriate frame transform when combining differently scaled or rotated views.

Drawable marks can be aggregated with other result graphics. Glyph spacing,
length and count are bounded controls. Graphics disabled skips drawing and both
optional previews, while retaining the numerical results and original image.
Synthetic accuracy checks do not establish performance on an actual inspected
surface; blur, noise, sampling and mixed orientations affect confidence.

## Attribution

The generator, image and project are original repository examples under the
repository's MIT license. No customer or third-party images are included.
