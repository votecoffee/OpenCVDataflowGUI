# Packed RAW10 / RAW12 sensor frames

Open `raw10.json` or `raw12.json` and **Run once**. The packed file includes a
short header and row padding. The verification block must report **maximum
absolute error 0** against the original ADC-count reference. A second branch
crops at an odd pixel origin, then demosaics using the captured Bayer phase.
The source and color-result previews stretch counts for viewing; the image
connections retain their uint16 sample values.

All buffers, images, projects and the generator are original synthetic content
under this repository's MIT license. Regenerate from the repository root with
`python "examples/opencv tools/packed_raw_sensor/generate.py"`.

**Packed RAW10 / RAW12 file** (`input.raw_sensor`) reads one configured frame.
**Unpack RAW10 / RAW12 sensor frame** (`cv.decode_raw_sensor`) accepts bytes,
bytearray or uint8 vectors/matrices from a plugin/user block. Convert SDK
memoryviews to bytes before connecting them, since graph history snapshots
cannot copy memoryview objects.

- `mipi_raw10` stores four high bytes followed by four low 2-bit fields in one
  byte. Width must be a multiple of four.
- `mipi_raw12` stores two high bytes followed by two low 4-bit fields in one
  byte. Width must be a multiple of two.
- These are the documented MIPI/V4L2 layouts. PFNC `Mono10p`/`Mono12p`, legacy
  vendor packing, 16-bit containers, compression and CSI packet headers require
  their own adapters; no packing is guessed from the filename or buffer.

Set decoded width/height, the byte offset to the first row, and row stride
including padding **on the final row**. Zero stride means tightly packed rows.
The default exact-size policy rejects missing or extra bytes; `at_least`
explicitly selects a frame prefix from a larger buffer/file. Frames are limited
to 16,777,216 decoded pixels and 128 MiB including offset/padding.

Right alignment preserves ADC counts on 0–1023 or 0–4095. Left alignment shifts
them into uint16; it does not stretch the maximum to 65535. Thresholds and other
inspection measurements should use those declared units. A separate diagnostic
preview affects viewing only.

Choose the sensor's Bayer tile and capture-origin row/column parity explicitly.
**Demosaic Bayer image → Use sensor metadata** reads this information, follows
integer crops, flips and right-angle rotations through the spatial frame, and
rejects resized/warped or already-consumed mosaics. The default manual Bayer
controls remain available. Monochrome frames do not support automatic Bayer
demosaicing. Bilinear/edge-aware modes accept these uint16 mosaics; VNG still
requires uint8.

Packed byte-grid coordinates are discarded. The optional `pixel_frame` input
can supply a calibrated decoded-image frame; otherwise the output gets a local
pixel frame. It does not infer the CFA tile origin. Tickets and other wire
payloads are retained; file sources can use their filenames as tickets.

Plugin adapters can supply the same optional `raw_sensor` metadata contract:
`model=raw_sensor_v1`, `mosaic=true`, a Bayer pattern, even/odd column and row
offsets, and the captured `pixel_spatial` frame. Acquisition details can be added
alongside those fields. Demosaicing marks the mosaic consumed. This lets an SDK
adapter supply an already unpacked image and reuse the phase-aware Bayer tool.

These are offline generated-fixture checks, not qualification of a camera SDK.
Packing references: [V4L2 RAW10](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-srggb10p.html)
and [V4L2 RAW12](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-srggb12p.html).
