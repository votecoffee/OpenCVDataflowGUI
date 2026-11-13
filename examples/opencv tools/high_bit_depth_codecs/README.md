# Lossless high-bit-depth codecs

Open `project.json` and **Run once** to compare PNG and JPEG 2000 encoding of
an original **193 × 129, 16-bit RGBA** inspection fixture. Both verification
blocks must report **maximum absolute error 0**, and both decoded outputs remain
uint16. `grayscale.json` runs the same checks on a grayscale input.

The asset, generator and projects are original content under this repository's
MIT license. Regenerate from the repository root with
`python "examples/opencv tools/high_bit_depth_codecs/generate.py"`.

**Encode image in memory** (`cv.encode_image`) now offers `jpeg2000`, using a
JP2 container. **Save image** uses the same controls for `.jp2` output. On the
supported OpenCV 5/OpenJPEG build, uint8 and uint16 grayscale, RGB/BGR and
RGBA/BGRA images preserve every sample in **lossless** mode, the default.
Dimensions and channel order stay unchanged, and encoded connections retain
their ticket, spatial frame and other payloads.

Choose **lossy** explicitly to use `jpeg2000_rate_x1000` (1–999). This is a
target rate, not a quality percentage or a guaranteed byte count. Lossy output
retains its storage depth but generally changes samples; the example's zero-error
verification will then fail. The lossless mode always sends 1000 to OpenCV.

This native JP2 encoder needs **at least 32 pixels in each dimension** with its
fixed resolution settings. Smaller images receive a validation error; PNG is
available for those sizes. Floating inputs require explicit conversion or a
floating-point format. A missing JPEG 2000 backend produces an unavailable-codec
error, without changing file type or creating an output. Existing files are
preserved. Raw J2K codestream output is not wrapped here.

PNG compression and strategy controls now include a filter dropdown (individual,
fast or all/adaptive filters) and the libpng zlib buffer size. Zero keeps the native
buffer default; explicit sizes range from 6 to 1,048,576 bytes. These settings
change encoding work and file size, while the sample data remains lossless.
The buffer setting does not apply to an SPNG backend. Bilevel PNG is still limited
to uint8 monochrome masks; use ordinary PNG for 16-bit data.

Use **Load image → pixel depth → unchanged** for file round trips. Graph payloads
remain on connections; they are not serialized into image files. This example
uses memory codecs and does not write files when run.

API reference: [OpenCV codec flags](https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html).
