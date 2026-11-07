# Stitch overlapping inspection views

Open [stitch_image_pair.json](stitch_image_pair.json) with **File → Open…**
and click **Run once**. The graph crops two 300 × 360 views from one
synthetic inspection image. **Stitch image pair** uses the recorded crop
coordinates to reconstruct the 480 × 360 image with 43,200 overlapping
pixels. **Measure image norm** checks the covered pixels against the
original and reports a maximum difference of zero.

Find the block under **Image processing → Geometry**. Image A defines the
mosaic's orientation and pixel scale. Use two views with a shared recorded
frame, enter an affine or projective matrix, or connect a transform from
**Estimate affine transform**, **Estimate image homography**, or
**Align images (ECC)**. The matrix must map **Image B → Image A**. For a
feature matcher, use Image B as Query and Image A as Train.

- **Coverage** marks every valid output pixel. Use it as a mask for measurements
  so uncovered pixels contribute no values. **Overlap** marks pixels supported
  by both views. **Layout** reports the mappings from both source grids to the
  mosaic, along with source frames, tickets, and pixel counts.
- Each source can have its own mask and region. Linear resampling requires all
  contributing Image B pixels to be valid. Prefer A, prefer B, or average
  samples in the overlap. Both images must have matching depth and encoding.
- **Min overlap pixels** controls acceptance. A failed overlap check still
  reports coverage and layout. **Anchor A** allows separate offline image
  tickets. The default requires matching tickets when both are present.

This workflow uses an existing registration or shared frame. Use the
[ordered scan example](../stitch_scan_sequence/README.md) to register a
batch of overlapping views automatically.
[OpenCV image transforms](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html)

```powershell
opencv-dataflow-run "examples/custom tools/stitch_image_pair/stitch_image_pair.json"
```

[All examples](../../README.md)
