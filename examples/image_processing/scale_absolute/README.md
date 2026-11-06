# Scale absolute to 8-bit

Open [scale_absolute.json](scale_absolute.json) with **File → Open…** and
click **Run once**. **Paired Sobel derivatives** produces the signed X
gradient. **Scale absolute to 8-bit** converts both gradient polarities
using a scale of 0.75. **Compare pixels** then marks magnitude values of
at least 120, highlighting strong transitions in either direction.

Find **Scale absolute to 8-bit** under **Image processing → Image math**.
It evaluates `abs(sample × scale + offset)` in the source's stored units
and rounds to uint8, with saturation at 255. The block preserves the
input image's dimensions, coordinate frame, and channel order. Every
stored channel is processed, including alpha on four-channel inputs.
Use grayscale or RGB/BGR-family images. [OpenCV conversion reference](https://docs.opencv.org/5.0/main_modules/core_array.html#convertscaleabs)

Enter **Scale** and **Offset** or wire numeric values to override them
for an activation. Optional mask and region inputs restrict the output
to their intersection. Pixels outside that selection are zero.

```powershell
opencv-dataflow-run examples/image_processing/scale_absolute/scale_absolute.json
```

[All examples](../../README.md)
