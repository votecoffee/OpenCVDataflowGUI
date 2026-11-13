# Build flat-field references from captures

Open `project.json` and run once. Three bright and three dark 16-bit RGB captures
are batched and averaged. **Build flat-field references** estimates the camera's
bright response and dark offset. **Flat-field correction** applies them to a
separate held-out image, removing shading while retaining a darker rectangle.
The final comparison must report zero maximum sample error.

`median.json` uses a localized outlier in the third bright capture. Median
estimation still matches the independent reference exactly. The held-out image
is never part of either calibration batch.

| Result | Expected value |
| --- | --- |
| Usable reference fraction | 100% |
| Corrected background, each RGB channel | 16384 counts |
| Corrected rectangle, each RGB channel | 8192 counts |
| Maximum error against direct reference pixels | 0 |

The fixture uses `observed = dark + response * reflectance`, with reflectance
`1/2` outside and `1/4` inside the rectangle. Correction targets half of integer
full scale, so the output is `reflectance * 32767.5`, rounded to uint16. The
expected PNG is assigned directly from these two constants; it is not generated
by the correction block. `assets/truth.json` records the geometry and values.

## Capture builder

`calibration.flat_field_reference` accepts **Bright** and optional **Dark**
FlowGroups from **Batch serial entries**. Use 1–64 captures per batch and choose
the minimum usable count per pixel (default three). Camera, exposure, gain,
lighting, optics and sensor pixel grid must remain compatible. The builder
does not register images or inspect hardware settings.
Use linear intensity samples; gamma-encoded data must be linearized first.

All captures must share dimensions, dtype and gray/RGB(A)/BGR(A) channel order.
Known grids must match by default. These independent saved files explicitly use
**Pixel grid** to declare that their pixel indices are already aligned. The
selected bright anchor defines output coordinates and pyramid context. Mask and
region selection always use that anchor grid, even in Pixel grid mode.

Choose mean or median per pixel. Even medians average the two middle accepted
values. Optional saturation rejection excludes a whole capture pixel if any
intensity channel reaches the configured ceiling. Alpha is omitted from all
calibration maps and retained unchanged on **Source**. Premultiplied captures
must be unassociated first.

The default outputs are float64 in **raw source sample units**, retaining
fractional estimates. uint16 captures therefore produce reference values in
integer counts, not normalized `[0,1]` values. Float32 is optional; **Same**
rounds integer estimates to nearest even. Validity is checked after publication
rounding/casting, so a response lost to quantization is not marked usable.

Unsigned-capture saturation, response and noise limits are fractions of full
scale. Floating-capture limits are raw sample values and may exceed one. The
response must be positive and at least **Minimum response** in every channel.
Enough accepted captures are required in both connected batches. Omitting Dark
explicitly assumes zero offset; its count map is then zero.

Optional noise checking uses population temporal standard deviation of accepted
samples about their mean, including in Median mode. It can reject noisy pixels.
**Retain noise** exposes bright/dark noise maps; when both options are off, noise
calculation is skipped. Mask and region select calibration pixels. Invalid or
unselected flat/dark/response pixels are zero, so they cannot be corrected later.
Selected invalid pixels appear in **Invalid mask**. At least one valid pixel and
the configured **Minimum valid fraction** are required for a passing builder.

Output payloads retain the anchor's coordinate/pyramid/companion context and
describe reference role and units. Training capture identities are listed in
Info. An outer batch ticket, when supplied, owns the derived result; a member
capture's ticket is retained only on Source and in the capture report.

Reduction uses bounded row tiles and per-pixel/channel scaling for large or tiny
floating samples. Complete reference maps are retained. Limits are 64 million
combined intensity samples, 4,194,304 pixels per image and 16384 per axis.

## Apply the references

Connect **Flat** and **Dark** to the existing correction block. Select
**Reference depth > Raw sample units** when the floating reference dtype differs
from the inspected image. The correction applies no automatic full-scale
conversion in this mode. Both projects preserve the held-out image's RGB order,
uint16 depth and filename ticket. Color references may omit an inspected
image's alpha channel; correction retains that image's original alpha.

**Response**, **Valid mask**, **Invalid mask**, accepted-count maps, noise maps
and Info are available in the builder's Results tab. Its optional response
preview does not change map values. Disabling graphics skips preview creation.
Use existing floating TIFF output blocks if references need to be saved for
another job; this example performs no file writes during execution.

## Attribution and regeneration

All sensor simulations, reference pixels, projects and code are original work
under this repository's MIT license. This tests arithmetic and graph contracts;
physical camera calibration needs its own captures and held-out qualification.

```powershell
.\.venv\Scripts\python.exe "examples/custom tools/flat_field_reference/generate.py"
```
