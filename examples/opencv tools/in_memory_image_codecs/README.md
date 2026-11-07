# In-memory image codecs

Open [in_memory_image_codecs.json](in_memory_image_codecs.json) with
**File → Open…**, then click **Run once**. The graph encodes the bundled
16-bit RGBA image as a PNG packet, decodes it, and verifies zero mean
absolute error. The encoder also reports the compressed byte count.

The blocks are under **Image processing → Codecs**.

- **Encode image in memory** produces a typed packet of compressed bytes.
  PNG supports unsigned 8-bit and 16-bit grayscale, color, and alpha.
  JPEG uses unsigned 8-bit grayscale or color and applies lossy compression.
  The codec settings are shared with **Save image**. Bilevel PNG accepts
  an unsigned 8-bit grayscale mask with values 0 and 255.
- **Decode image from memory** restores the original channel order and
  stored depth by default. The inspector can select a different output
  encoding or pixel depth. The source ticket and pixel frame travel with
  the packet.

Both blocks use OpenCV's single-image codec path. Packets are limited to
128 MiB and 16,777,216 image pixels. [OpenCV codec flags](https://docs.opencv.org/5.0/main_modules/imgcodecs_flags.html)

```powershell
opencv-dataflow-run "examples/opencv tools/in_memory_image_codecs/in_memory_image_codecs.json"
```

[All examples](../../README.md)
