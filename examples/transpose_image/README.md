# Transpose image

Open [transpose_image.json](transpose_image.json) with **File → Open…**
and click **Run once**. The graph crops a 160 × 200 patch and reflects
it across its anti-diagonal. **Transform image points** maps three
points from the resulting 200 × 160 image back to the full source.
The output points are `[[414, 279], [255, 80], [334, 248]]`.

Find **Transpose image** under **Image processing → Geometry**.
**Main** swaps X and Y. **Anti** swaps the axes and reverses both
directions. Both preserve pixel values and channel encoding. The
block also accepts 32-bit signed label planes and physical depth
with its units and missing samples intact. [OpenCV transpose reference](https://docs.opencv.org/5.0/main_modules/core_array.html#transpose)

**Geometry** maps output pixel centers into the immediate input.
The image payload also composes this mapping with earlier operations.
The example uses **Read port payload / coordinates** to get the
composed mapping to the original image.

```powershell
opencv-dataflow-run examples/transpose_image/transpose_image.json
```

[All examples](../README.md)
