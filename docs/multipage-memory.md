# Multipage TIFF buffers

Use **OpenCV → Image Codecs → Codecs** to encode an ordered image group with
`cv.encode_pages` and decode it with `cv.decode_pages`. **Batch serial entries**
creates a group; **Unbatch serial entries** emits the decoded pages individually.
The existing TIFF file blocks remain available for recorded acquisition and saving.

For random file-page requests beyond a 64-image group, use
[Read TIFF page on demand](tiff-page-access.md), with joint page/byte cache limits,
snapshot refresh policies and a wired integer selector. It reads up to 1024
inspected TIFF pages individually without retaining every decoded page.

## Pixels and page context

**Encoded** carries an immutable `EncodedPages` packet containing TIFF bytes and
a bounded JSON sidecar. Each page retains its label, logical color order, ticket,
outcome, timing, coordinates, pyramid information and arbitrary JSON payload.
Image-attached metadata and connection payloads are merged using the normal
payload rules. Descriptors must match the encoded size, depth and channels.
Sidecars are private copies; changing a returned context cannot alter the packet.
These descriptors provide consistency checks, not authentication of a capture.

**Buffer** contains ordinary TIFF bytes. It does not embed the application's
per-page ticket, transform, label or recipe payload. The decoder also accepts
uint8 vectors. Raw buffers use native gray/BGR/BGRA order, inherit the incoming
container payload and receive a distinct fresh coordinate frame per page.
Stale image-grid/pyramid/raw-sensor/alpha context is cleared. A container ticket
does not recover the original individual acquisition tickets.

The decoder outputs an ordered **Pages** group, **Page count**, an **Info** report
and an **Image** preview. **Preview page** addresses a position in the selected
subset. Its image carries that page's payload. **Start page** is zero based;
**Stride** selects every nth page; **Page count** zero selects all remaining
strided pages. A positive count caps the available selection. No new source
ticket is issued by these processing blocks.

## Controls and limits

- Gray, RGB/BGR and RGBA/BGRA pages may have different sizes and depths:
  uint8, uint16, float32 or float64. Finite signed HDR values are supported.
  Physical `DepthArray` planes require depth-specific tools instead.
- None, LZW, Deflate and PackBits compression are dropdown choices. Explicit
  compression avoids native float-color LogLuv defaults. Horizontal prediction
  requires integer pages; floating prediction requires floating pages. Explicit
  predictors require LZW/Deflate. Mixed integer/float groups use Native prediction.
  Info distinguishes requested compression from the actual mode of each page;
  the installed native encoder stores float64 pages uncompressed.
- Strip height zero uses the native default. Both DPI values must be set, or
  both zero. DPI tags do not establish calibrated physical coordinates.
- Encoding verifies every page by native readback, including exact sample bytes,
  before publishing either output. This check also runs in Lean mode and costs
  an additional decode. An unavailable or lossy native combination fails visibly.
- Unchanged output format/depth preserves each typed page's logical encoding and
  precision. Explicit conversions use the existing full-scale mapping. HSV/Lab
  requires uint8 or float32. Conversion does not change the page geometry.

Buffers and uncompressed input groups are limited to 128 MiB; sidecars to 2 MiB.
Each page is limited to 16,384 per axis and 16,777,216 pixels. At most 64 selected
pages travel together, from a container with at most 1,024 primary directories.
Classic TIFF and BigTIFF, both byte orders, are supported for top-left contiguous
gray/RGB(A) pages with the depths/compressions above. Palette, bilevel, CMYK,
signed-integer, rotated, planar and other compressed TIFF variants are rejected
by these memory blocks; this does not narrow the existing file reader.

Before native pixel decoding, directory traversal checks field/strip/tile offsets,
counts, dimensions and cycles. The total selected pixel budget and a conservative
sample/output/conversion workspace estimate must fit the configured limits.
Info reports the estimate. Strided pages are decoded individually; unselected
pages are not materialized. Encoded bytes, graph snapshots, Python object overhead
and codec internals are outside that workspace estimate. These checks are not
process isolation for hostile native files.

Qualification uses the installed OpenCV 5 TIFF codec. Float64 multichannel TIFF
can emit a native photometric/ExtraSamples warning despite exact native sample
readback; cross-reader metadata interoperability remains unqualified. Independent
Pillow readback covers integer gray/RGB pages and DPI. This increment does not
qualify OpenEXR or claim support for every TIFF variant.

[Five runnable projects](<../examples/opencv tools/multipage_memory/README.md>) ·
[Native OpenCV APIs](https://docs.opencv.org/5.0/main_modules/imgcodecs.html) ·
[Image formats](image-formats.md) · [Tickets and groups](control-flow.md)
