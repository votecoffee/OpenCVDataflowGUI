# Multipage TIFF output

Open [multipage_tiff_output.json](multipage_tiff_output.json) with
**File → Open…**. Start continuous execution or click **Run once** three
times. The source replays three pages. **Batch serial entries** groups
them in order and **Write multipage TIFF** creates `tiff_export.tiff` in
the working directory. You can set an explicit new output path in the
writer's inspector.

**Write multipage TIFF** is under **Output → Images**. Connect an
ordered group of up to 64 images. The writer accepts unsigned 8-bit and
16-bit grayscale and RGB/BGR images, including alpha. Pages may differ
in size and depth. Choose uncompressed, LZW, Deflate, or PackBits
storage. LZW and Deflate also support horizontal prediction. Set both
DPI fields to store a print resolution, or leave both zero for the codec
defaults.

The writer creates one file per complete batch. Use a new path for
another export. It reports the written page count and byte count after
publication. The raw batch and encoded file each have a 128 MiB limit.
[OpenCV multipage encoding reference](https://docs.opencv.org/5.0/main_modules/imgcodecs.html#imencodemulti)

```powershell
opencv-dataflow-run "examples/opencv tools/multipage_tiff_output/multipage_tiff_output.json" --all-images
```

[All examples](../../README.md)
