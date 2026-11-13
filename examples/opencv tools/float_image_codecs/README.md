# Floating-point image codecs

Open `project.json` and **Run once**. It reads an original float32 color TIFF
whose samples exceed one. The TIFF encode/decode branch must report **maximum
absolute error 0**. The Radiance HDR branch must pass a **0.0625** error limit
for this fixture, while showing a nonzero error from RGBE quantization.
`float64.json` separately verifies exact float64 RGBA TIFF storage, including
negative values and channel precision beyond float32. On the current build,
multichannel float64 TIFF decoding emits a native photometric/ExtraSamples
warning. Samples and channel order pass exact OpenCV readback checks; TIFF
metadata interpretation by other readers remains unqualified.

The fixtures, projects and generator are original content under the repository's
MIT license. Regenerate from the repository root with
`python "examples/opencv tools/float_image_codecs/generate.py"`.

**Encode image in memory** and **Save image** share their codec controls:

- TIFF supports uint8/uint16 and float32/float64 gray or color/alpha images.
  Explicit lossless compression defaults to none, avoiding the native float32
  color LogLuv default. LZW, deflate, packbits, predictors and strip height are
  configurable. Floating prediction requires float samples; horizontal
  prediction requires integer samples. Predictors apply to LZW/deflate.
- Radiance HDR requires nonnegative float32 RGB/BGR without alpha and samples
  below `2**127`. RLE and uncompressed modes both use lossy shared-exponent
  RGBE values. Dim channels can lose precision beside bright channels, and
  values below approximately `1e-32` can become zero. Use TIFF for signed or
  exact floating data.
- EXR configuration is available for a capable OpenCV build, with float32 or
  quantized float16 storage and compression controls. **The current OpenCV 5
  build reports OpenEXR: NO**, so EXR execution fails clearly without creating
  a file. Native EXR round-trip qualification remains open; the app does not
  enable codecs through global environment changes.

**Decode image from memory** restores the packet's channel order. Use
`pixel_depth=unchanged` in **Load image** to preserve file samples. Float64
RGB/RGBA conversion retains double precision; float64 HSV/Lab conversion still
requires an explicit supported depth. Multipage TIFF output/replay also supports
floating pages. Image payloads, including frames and tickets, stay on encoded
connections; graph payloads are not serialized into the external image file.

The normalization node is a separate display branch. Storage and verification
use the original sample values. Existing image files are preserved by the saver;
choose a new output path when testing file output. PNG/JPEG reject unsupported
floating input rather than silently converting it to 8-bit.

API reference: [OpenCV image codecs](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html).
