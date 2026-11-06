# Raw YUV decoding

Open [raw_yuv.json](raw_yuv.json) with **File → Open…**, then click **Run once**.
The two branches decode the same 120 × 64 NV12 color bars. One reads the
raw frame file. The other loads the identical packed byte matrix from a
grayscale PNG and decodes it inside the graph.

- **Input → Files → Raw YUV frame** reads one tightly packed frame. Set the
  decoded width, height, and layout. The source checks the exact byte count
  and can use the filename as its ticket.
- **Image processing → Color and depth → Decode packed YUV** accepts a
  two-dimensional uint8 matrix. For 4:2:0, use `(height × 3 / 2, width)`.
  For 4:2:2, use `(height, width × 2)`. The result has a decoded pixel frame
  and keeps the input ticket.

Both blocks support NV12, NV21, I420, YV12, YUY2, UYVY, and YVYU. Width
must be even, and 4:2:0 also needs an even height. Color conversion uses
OpenCV's limited-range BT.601 coefficients. **Gray** returns the stored
luma samples. RGB, BGR, RGBA, and BGRA outputs are also available.
[OpenCV subsampled YUV conversion reference](https://docs.opencv.org/5.0/main_modules/imgproc_color_conversions.html)

```powershell
opencv-dataflow-run examples/inputs/raw_yuv/raw_yuv.json
```

[All examples](../../README.md)
