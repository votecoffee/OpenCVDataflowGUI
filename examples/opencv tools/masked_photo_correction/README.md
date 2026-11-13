# Masked illumination and texture correction

Open [masked_photo_correction.json](masked_photo_correction.json), choose the
**Debug** profile, and run once. Two branches share an editable rectangular
region on the same original synthetic surface. No download or plugin is needed.

- **Change selected illumination** exposes native `alpha` and `beta` controls.
- **Flatten selected texture** exposes both Canny thresholds and a Sobel aperture
  dropdown. Raise thresholds to suppress weaker texture while retaining strong edges.
- Inspect **Image**, **Selected mask**, **Gradient mask**, **Change** and **Report**.
  The change image is the maximum absolute color-channel difference in 8-bit DN.
  The graphic layer marks edit bounds, rather than claiming those bounds are a defect.

The source has a designed vertical step at **x=96**, a broad intensity bump and
weak periodic texture. Independent measurements on the bundled reference give:

| Measurement | Input | Illumination edit | Texture edit |
| --- | ---: | ---: | ---: |
| Selected left-side profile range, DN | 56.58 | 36.10 | 10.63 |
| High-pass texture standard deviation, DN | 2.54 | 0.71 | 0.28 |
| Step position, pixels | 96 | 96 | 96 |
| Step difference in the averaged profile, DN | 63.21 | 6.69 | 46.19 |

These are controlled image measurements, not calibrated physical glare or defect
measurements. In particular, illumination editing retains the boundary location
but greatly reduces its contrast. It does not recover clipped samples or replace
flat-field calibration. Check the downstream inspection against original inputs.

## Selection and data contract

Use uint8 BGR/RGB, optionally with straight alpha. Channel order, alpha, image
coordinates, ticket and attached payload data are retained. Integer 16-bit and
floating input require an explicit conversion; these tools do not silently
quantize them. Premultiplied alpha must first be unassociated. Images must be
at least 8×8 and no larger than 2.1 million pixels for the full-image Poisson solve.

An input mask and region are intersected. Circles, rectangles and polygons use
the shared region tools. The native solver receives that selection and erodes it
by three pixels for its gradient weights; **Gradient mask** exposes those weights,
not a guarantee that only those interior pixels change. The full source supplies
the boundary conditions. Only original selected pixels receive the solved image;
all other samples, including holes, remain byte-exact. Alpha is never edited.
An empty selection or one with no gradient interior returns the original image
and an explicit no-op reason, without invoking the solver.

OpenCV uses the native Canny defaults 30/45 even though the API prose describes
a different typical threshold range. This wrapper accepts ordered, finite
thresholds from 0 to 65535 and the native apertures 3/5/7. This example uses 50/150.

## Source and reproduction

The fixture, graph and independent limits in [truth.json](truth.json) are original
project work under MIT. [build_example.py](build_example.py) regenerates them.
No customer, tutorial or third-party image content was used.

```powershell
python "examples/opencv tools/masked_photo_correction/build_example.py"
opencv-dataflow-run "examples/opencv tools/masked_photo_correction/masked_photo_correction.json" --profile debug
```

Native API: [OpenCV photo operations](https://docs.opencv.org/4.x/df/da0/group__photo__clone.html).
The installed OpenCV 5 binding reference is also available in the block's documentation editor.

[All OpenCV examples](../README.md) · [Example data policy](../../README.md#image-policy)
