# Lateral chromatic aberration correction

Three built-in blocks align the red and blue channels to the unchanged green
pixel grid using OpenCV 5's polynomial correction. Find them under **OpenCV →
Computational Photography (photo)**. No additional Python dependency is needed.

## Calibration models

**Load chromatic calibration** (`input.chromatic_calibration`) reads OpenCV
UTF-8 YAML, XML or JSON through `loadChromaticAberrationParams`. The file has
integer `image_width` and `image_height`, and `blue_channel`/`red_channel` mappings,
each containing `coeffs_x` and `coeffs_y` sequences. Polynomial degree is inferred
from the triangular coefficient count. An optional Node field selects a named
root mapping. Relative paths resolve beside the saved project or working
directory. Files are bounded to 256 KiB; malformed files fail without emissions.

**Define chromatic calibration** (`calibration.chromatic_model`) exposes the
image dimensions, degree and coefficient matrix for saved or synthetic models.
The four rows are **blue dx, blue dy, red dx, red dy**. For degree d, each row
contains `(d+1)*(d+2)/2` entries. Terms are ordered:

```text
1, v, u, v², uv, u², v³, uv², u²v, u³, ...
u = (x - width/2) / (width/2)
v = (y - height/2) / (height/2)
input donor = (output x - dx, output y - dy)
```

Coefficients are stored as native float32. Models carry a content fingerprint
covering coefficients, grid, degree and optional camera/lens labels. Modified
models with stale fingerprints are rejected. A model travels as ordinary data
and can be supplied by a recipe or a user block through the same schema.
These blocks load or define a supplied model; they do not estimate a camera's
coefficients. OpenCV's calibration application fits black-disc target images.
Use captures made with the same camera, lens, focus and aperture as calibration;
camera/lens labels are descriptive and do not authenticate hardware.

## Applying correction

**Correct chromatic aberration** (`cv.correct_chromatic_aberration`) requires
image and calibration inputs. The image dimensions must exactly match the
calibration. Default Original grid policy also rejects known crop, resize,
rotation and pyramid transforms. Current grid explicitly permits a calibration
made for that exact transformed grid. The block never silently rescales a model.

RGB/BGR and straight-alpha RGBA/BGRA inputs retain their encoding, depth and
alpha samples. Supported color depths are uint8, uint16, int16, float32 and
float64. HSV/Lab and physical depth planes require an appropriate explicit
conversion first. Red/blue values are remapped by native linear interpolation;
green defines the coordinate reference. Existing frame/pyramid metadata, ticket,
acquisition start time and companion payloads remain attached to that grid.
The payload records the calibration fingerprint under `chromatic_correction`.
This is channel alignment; it does not replace ordinary lens undistortion or
establish a physical coordinate calibration.

An explicitly selected Bayer pattern enables native bilinear demosaicing of
uint8/uint16 mosaics into BGR. The pattern describes the current top-left 2×2
tile; known sensor phase must agree. Bayer border samples synthesized by
demosaicing are excluded from measured support. Use the separate Demosaic Bayer
image block first for other algorithms or additional sensor controls.

Mask and region inputs select **output pixels to correct**, intersecting when
both are wired. Native correction still uses the whole source grid and can
sample outside that selection. Unselected color pixels retain their original
values. For Bayer input they retain the uncorrected demosaiced color values.

Outputs include:

- **Image:** corrected pixels, withheld if selected support fails the limit.
- **Valid mask:** selected pixels whose red/blue interpolation footprints are
  supported by the original image. Bayer additionally requires interior support
  for green and the demosaicing neighborhoods.
- **Processed mask:** requested output selection, including unsupported borders.
- **Valid fraction:** supported pixels divided by selected pixels.
- **Report:** grid/degree/fingerprint, selected and supported counts, maximum
  per-channel displacement, fixed native sampling settings and limit outcome.

Native sampling replicates image borders. The validity mask excludes those
extrapolated samples conservatively, including fractional donors outside the
image even when native 1/32-pixel quantization rounds them onto an edge.
Invalid border → Zero clears unsupported corrected RGB pixels while preserving
alpha. Both policies leave unselected pixels alone. Minimum valid fraction
defaults to 0.5; a lower result emits failed diagnostic masks/report and no
primary corrected image event. It does not assign a final ticket judgment.

Support evaluation uses bounded strips. Controls cap displacement and
`pixels * coefficient_terms * 4` work before native image correction. The hard
image limit is 2.1M pixels, 16384 per axis, with degree 0–11. Coefficients must
be finite and within ±1e6. Raising work/displacement limits does not validate a
bad optical calibration or prove model invertibility.

Optional drawable support text obeys the project's graphics/run profile.
Higher-depth ASCII text and labels use the existing sample-preserving font
renderer, avoiding OpenCV 5's uint8-only text restriction. Default uint8 ASCII
annotations retain their existing Hershey rendering.

## Example and validation

Open the [original 16-bit fringe example](<../examples/opencv tools/chromatic_aberration/README.md>).
It corrects independently generated blue (-2,+1) and red (+3,-2) displacements,
preserves all supported reference pixels exactly and reports the known border
loss. A second project defines the same model in the inspector. A third
deliberately requires 100% support and demonstrates failure with no image event.

Focused tests also check quadratic and degree-11 conventions, all supported
color depths/orders, alpha, selected processing, Bayer patterns, OpenCV file
formats, transformed grids, ticket/timing/pyramid ownership, work guards,
malformed data, XML persistence and depth-preserving result composition.
Synthetic correction verifies the supplied-model path; real-lens fit quality
and live camera settings require their own captured calibration evidence.

[OpenCV API](https://docs.opencv.org/5.0/main_modules/photo_ca_correction.html)
· [OpenCV calibration tutorial](https://docs.opencv.org/5.0/py_tutorials/py_photo/py_chromatic_aberration/py_chromatic_aberration.html)
· [OpenCV 5.0.0 implementation convention](https://github.com/opencv/opencv/blob/5.0.0/modules/photo/src/chromatic_aberration_correction.cpp)
