# Image metadata round trips

Open a project and select **Run**. These original tool fixtures demonstrate
metadata on explicit connections alongside image data.

- `png_roundtrip.json`: read the tagged source, encode/decode it twice, and retain
  every RGBA16 sample and EXIF/XMP/ICC/CICP chunk.
- `jpeg_roundtrip.json`: explicitly load RGB8 pixels, create fresh metadata, and
  round-trip EXIF/XMP/ICC. JPEG pixels are lossy; metadata is checked exactly.
- `oriented_once.json`: apply EXIF orientation 6, create fresh normal-orientation
  metadata, and encode/decode again. The final image matches
  `oriented_reference.png`, with its source ticket and coordinate transform.
- `unsupported_jpeg_cicp.json`: intentionally fails because JPEG does not support
  CICP. It emits no encoded buffer and its downstream decoder does not execute.

Select a decoder and open its **Results** tab to inspect its image, metadata packet
and `info` output. No example writes output files when run. The installed CLI
checks each project in Debug and Lean; the unsupported case exits with failure.

## Using the blocks

**Create image metadata**, **Encode image in memory** and **Decode image from
memory** are under **OpenCV → Image Codecs (imgcodecs) → Codecs**.

For an existing file, set **Image file → Metadata mode** to `read` and expose its
optional `metadata` output. Connect that output to the optional `metadata` input
of **Encode image in memory** or **Save image**. The ports are hidden by default.
Metadata mode `off` retains the existing pixel-only behavior. Decoders can report
chunks without applying orientation, or apply EXIF orientation explicitly.

**Create image metadata** creates fresh EXIF orientation/software/description
fields, optional UTF-8 XMP XML, an ICC profile from a file, and optional CICP bytes.
Disable **Include EXIF** for other chunk types alone. EXIF text is ASCII; XMP
supports Unicode. Orientation choices use a dropdown. All existing JPEG/PNG
compression controls remain available on the encoder and file writer.

**Save image** uses the native metadata file API when metadata is connected. Set
its path to a new file before running. Existing files are preserved. Native output
is staged and decoded for verification before the destination is created; failed
metadata verification removes the staging file and leaves no destination.

## Metadata and orientation contract

The immutable `ImageMetadata` packet stores at most four distinct native chunk
types, at most 1 MiB each. PNG supports EXIF, XMP, ICC and CICP; JPEG supports
EXIF, XMP and ICC. Other metadata-write formats and unknown/duplicate chunks are
rejected explicitly. ICC color space must match the stored gray/RGB pixels;
tagging a profile does not perform color conversion.

The wrapper handles JPEG's XMP namespace header and PNG's string terminator at
the native boundary. Graph packets contain canonical TIFF EXIF, XML text and raw
ICC/CICP bytes. Requested chunks must survive readback before a buffer or file
is emitted. See the [OpenCV codec API](https://docs.opencv.org/5.0/main_modules/imgcodecs.html).

Writing an orientation tag does not rotate stored pixels. Explicit orientation
updates the decoded image's coordinate/pyramid transform and preserves its ticket.
The original raw metadata is marked read-only after orientation: it can contain
stale orientation, dimensions or thumbnails. Use fresh metadata for transformed
pixels, as `oriented_once.json` demonstrates. Different stored sizes or known
coordinate frames are rejected. Opaque camera tags are preserved as bytes; this
is not a general editor that repairs arbitrary tags after image processing.

## Fixture and attribution

`build_example.py` creates the original MIT-licensed pixel ramps, tagged source,
reference image and graphs. No customer or third-party images are included.
Each graph includes an attribution/scope comment. `truth.json` records the known
sizes, orientation and expected chunk types.

The sRGB profile is generated with Pillow/LittleCMS. Its embedded copyright text
is **“No copyright, use freely”**. The generator fixes the header creation date
and clears the optional profile ID to make fixture bytes reproducible. The
profile is used for metadata transport tests, not color-calibration claims.
