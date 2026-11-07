# Subpixel patch extraction

Open [subpixel_crop.json](subpixel_crop.json) with **File → Open…** and
click **Run once**. The template matcher finds a target and refines its
center. **Crop subpixel patch** samples a 37 × 31 image around that
fractional point. The final block maps the patch center back to the
scene using the **Geometry** output.
The example rejects failed match centers. If the crop emits no geometry,
the connected point transform waits.

Find **Crop subpixel patch** under **Image processing → Geometry**.
Use the image center, enter a custom center, or wire a single point
in image coordinates. A wired **Pose** uses its origin, resolved in
the input image's coordinate frame. The patch axes stay aligned with
the image axes. Width and height count output pixels, including when
the center falls between pixel centers.

**Strict** requires all patch sample centers to fit in the source.
**Replicate** extends edge pixels and emits a patch-aligned **Valid mask**
that marks centers inside the source. Both modes require the requested
center to lie between the outermost input pixel centers. Sampling uses
bilinear interpolation. [OpenCV subpixel sampling reference](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html#getrectsubpix)

The block accepts uint8, uint16, int16, and float32 grayscale and color
images, including alpha channels. **Same** preserves the input depth
and encoding. **Float32** retains stored sample units, such as 0–255
for promoted uint8 values. Changing HSV or Lab depth requires an
explicit color conversion first.

```powershell
opencv-dataflow-run "examples/opencv tools/subpixel_crop/subpixel_crop.json"
```

[All examples](../../README.md)
