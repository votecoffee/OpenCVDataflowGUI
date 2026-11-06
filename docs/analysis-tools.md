# Analysis tools and parameter conventions

All declared enum parameters appear as dropdowns in the right inspector. Booleans
are checkboxes; vector and matrix controls accept JSON arrays. Optional image
arguments such as masks and coordinate maps are connectable ports. Validation runs
before configuration is applied and before execution. Unknown enum values from an
older/custom project remain visible and produce a configuration error.

The [parameter reference](block-parameters.md) is generated from live block
metadata, including defaults, choices, optional ports and control help. Regenerate
it with `python scripts/block_parameter_reference.py`; use `--check` to verify it.
New defaults are merged into older project configurations when loaded. Existing
parameter names and default behavior are retained.

## Added tools

| Block | Inputs and results | Controls |
| --- | --- | --- |
| Box filter | Image → image | Independent width/height, normalization, anchor, output depth, border |
| Separable 2D filter | Image → image | X/Y coefficient vectors, anchor, delta, output depth, border |
| Harris / minimum eigenvalue response | Image → floating response and display preview | Method, block size, derivative aperture, Harris K, border |
| Refine corners to subpixel | Image and point list → refined point list | X/Y window, zero zone, termination mode, iteration limit, epsilon |
| Phase correlation shift | Two images and optional floating window → shift and response | Float precision, generated Hanning window, acceptance response |
| Iterative phase correlation | Reference and moving images → subpixel shift, inverse warp, and status | Float precision, odd refinement neighborhood, iteration limit |
| Estimate image translation | Matched image points → translation matrix, inliers, and residuals | RANSAC/LMeDS, pixel tolerance, iterations, confidence, refinement and acceptance limits |
| Estimate affine transform | MatchORB correspondences → transform and inlier count | Full/partial model, RANSAC/LMedS, reprojection threshold, iterations, confidence, refinement and acceptance limits |
| Remap image coordinates | Image and float32 X/Y maps → image | Interpolation, border and constant fill |
| Pack remap coordinates | Float32 X/Y maps → fixed-point packet | Subpixel fractions or rounded nearest coordinates |
| Remap with packed coordinates | Image and fixed-point packet → image | Compatible interpolation, border and constant fill |
| Cartesian to polar fields | Matching monochrome float X/Y fields → magnitude and angle | Radians/degrees |

Connect **Good features to track > points** to **Refine corners > points**, with
the same image connected to both. Corner response exposes the raw float response
for downstream processing and a separate normalized image for display.

Phase correlation reports the translation from image A to image B in pixels.
Its response output retains the native number, while the node's score is clamped
to [0,1] for display. A connected window takes precedence over the generated
Hanning window and must match the chosen float precision and image dimensions.

[Iterative phase correlation](../examples/iterative_phase_correlation/README.md)
uses OpenCV's built-in Hann window and returns an inverse translation for
**Warp affine**. Use matching even-sized inputs with distinctive detail. The
block withholds the shift and transform for flat interiors and invalid or
half-period shifts.

[Estimate image translation](../examples/translation_registration/README.md)
fits only X/Y displacement from matched feature points. The accepted transform
maps query to train. The fit preserves each pair's inlier label and residual.
Connected affine and perspective warp transform inputs require an emitted value.
Unconnected transform inputs use the configured matrix.

Affine estimation consumes the same query/train point dictionary as homography
estimation. The transform maps query pixels to train pixels and can connect to
**Warp affine > transform** or **Transform points > transform**. The partial model
uses rotation, uniform scale and translation; the full model also permits shear
and independent axis scaling. These are image-space transforms, separate from
grid-based physical calibration.

Remap accepts two single-channel float32 coordinate images. Each output pixel
samples the source at `(map_x, map_y)`. **Pack remap coordinates** converts
the pair for **Remap with packed coordinates**. Both paths preserve the
rectified frame when given lens correction maps. The
[packed remap example](../examples/packed_remap/README.md) compares their
pixel values. Cartesian-to-polar inputs must share dimensions and floating
depth. The units describe the angle output only.

## Expanded existing controls

- Gaussian kernels can be rectangular, with separate X/Y sigma and border controls.
  Ordinary and Advanced G-API Canny expose aperture and L2 gradient calculation.
- Filters, gradients, morphology and pyramids expose relevant anchors, borders,
  dimensions and output depths. `reflect` retains the historical REFLECT_101
  behavior; `reflect_edge` selects BORDER_REFLECT. Unsupported native depth/border
  combinations report an execution error rather than silently substituting one.
- Threshold exposes all five basic modes and output maximum. Automatic and
  adaptive thresholds expose their maximum too. Contours expose retrieval,
  approximation and a coordinate offset.
- ORB exposes first level, WTA_K, score type and patch size alongside its existing
  controls. Masks are optional ports on supported detectors and image operations.
  Descriptor matching automatically chooses the Hamming norm for the ORB WTA_K,
  rejects incompatible explicit norms and supports ratio or cross-check matching.
- Homography exposes its solver, confidence and iteration limit. Calibration
  exposes SB/classic grid detection, refinement, lens-model flags, solver and
  termination controls. Initial camera parameters are required for selected fixed
  or intrinsic-guess modes; an optional new camera matrix controls undistortion.
- Camera acquisition exposes backend and requested width, height, FPS and buffer
  size. Zero retains backend defaults; setting a request cannot guarantee hardware
  support. No physical camera is part of automated validation.
- Image saving exposes JPEG quality/progressive/optimization/restart controls,
  PNG compression/strategy/bilevel controls and WebP quality. Controls apply only
  to their codec, and native defaults are preserved when left unset. Existing
  files remain protected against overwrite.

These wrappers use eager OpenCV calls except the explicitly named G-API tools.
Native destination buffers and internal pointers are runtime-managed. The catalog
does not claim coverage of every OpenCV algorithm or backend-specific codec flag.

Native API details: [image filtering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html),
[feature detection](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html),
[motion estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), and
[geometric transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).
