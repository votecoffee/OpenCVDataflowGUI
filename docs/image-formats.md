# Image-file conversion and image encodings

The Image file block (`input.file`) decodes using OpenCV's `IMREAD_UNCHANGED`, then
converts pixel depth and finally color layout. This retains decoded source depth
and alpha until an explicit conversion discards them. File paths support Unicode.
The decoded source is retained as a diagnostic graphic. Source files are read only.
Available codecs depend on the installed OpenCV build. This is pixel conversion,
not ICC profile management, Bayer demosaicing, EXIF autorotation, or raw camera decoding.

`path_base` defaults to `working_directory` for existing jobs. Select `project` to
resolve a relative image path against the saved root project JSON's directory.
The path remains relative in saved files and survives worker execution and nested
ToolBlocks. Save/open a project before using that mode; when moving a project or
saving it elsewhere, move/copy its image assets too. This option currently applies
to Image file inputs; other file blocks retain their existing path behavior.

## Settings

| Setting | Choices | Default |
| --- | --- | --- |
| Color format | unchanged, bgr, rgb, bgra, rgba, gray, hsv, lab | bgr |
| Pixel depth | unchanged, uint8, uint16, float32 | uint8 |

The depth is per channel: uint16 BGR has three 16-bit channels, not 16 bits total.
The `unchanged` choices preserve the decoded representation, normally GRAY, BGR,
or BGRA. Alpha is preserved for unchanged/BGRA/RGBA output and dropped for other
formats. Newly added alpha is fully opaque. Other than color layout and depth,
image dimensions are unchanged.

## Numeric rules

- uint8 and uint16 have full scales of 255 and 65535. Conversion scales by the
  ratio of full scales and rounds to nearest with NumPy's ties-to-even rule.
- Integer-to-float32 divides by the source full scale. A white uint16 sample
  becomes 1.0, rather than 65535.0.
- Floating source values are preserved when retaining depth or selecting float32.
  Color conversion assumes OpenCV's usual normalized BGR/GRAY float convention.
- Float-to-integer saturates to [0,1], scales to the target full scale, and rounds.
  It never wraps negative or over-range samples into unrelated values.
- Nonfinite samples and unsupported source depths fail explicitly. Unchanged depth
  can retain float64 grayscale/BGR images, but color conversion requires uint8,
  uint16, or float32. Selecting float32 permits conversion of float64 sources.
- HSV and Lab support only uint8 and float32. uint8 HSV has hue in [0,179] and
  saturation/value in [0,255]; float HSV has hue in [0,360] and normalized
  saturation/value. uint8 Lab uses OpenCV's encoded L/a/b representation;
  float Lab uses L in [0,100] with signed a/b coordinates. These are not uniformly
  normalized color channels. uint16 HSV/Lab requests fail instead of losing depth.

Conversions are deterministic across images: a constant 16-bit mid-gray image
previews as mid-gray, and does not become black through per-image min/max scaling.
HSV8 and Lab8 round trips can lose precision through color-space quantization.

## Image contract and downstream tools

`ImageArray` is an `np.ndarray` subclass carrying `color_format`. Standard block
snapshots and ToolBlock result copying preserve that attribute. Ordinary untagged
arrays retain the existing OpenCV conventions: 2D is grayscale, 3 channels are BGR,
and 4 channels are BGRA. `info` is a structured output with width, height, channels,
color format, pixel depth, bits per channel, and decoded source format/depth.

The Qt preview converts tagged encodings to RGB for display. Built-in G-API tools,
thresholding, contour inspection and image saving use `opencv_image(image)` to
interpret RGB/RGBA/HSV/Lab before processing. This canonicalizes color encoding,
without silently converting pixel depth. Native G-API Grayscale and Canny currently
require uint8. Other native tools may also limit channel layouts or depths; a typed
image wire does not promise that every operation accepts every image representation.

Custom blocks can import helpers from `opencv_dataflow.images`. Call
`opencv_image(image)` when a native operation expects OpenCV channel order. If an
operation returns pixels in a nonstandard encoding, tag them explicitly with
`ImageArray(result, "rgb")` (or the actual encoding). OpenCV operations normally
return plain arrays, so metadata must be reapplied if the output remains HSV/RGB/etc.
Array views that change the channel layout also require an appropriate new tag.

Saving uses canonical OpenCV channel order. The resulting file's supported depth,
alpha and compression behavior are still determined by the selected codec; use
PNG for lossless uint8/uint16 or a suitable TIFF format for floating-point data.

References: [OpenCV decoding](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html)
and [color conversions](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html).
