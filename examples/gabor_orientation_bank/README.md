# Gabor orientation bank

Open [gabor_orientation_bank.json](gabor_orientation_bank.json) with
**File → Open…** and click **Run once**. The bank tests 0°, 45°, 90°,
and 135° on three patches. The interior of the vertical stripes
reports 0°, the horizontal stripes report 90°, and the flat patch
has no valid orientation. Near patch boundaries, the filter window
can include more than one texture.

**Gabor orientation bank** is under **Image processing → Filtering**.
Enter 2–16 distinct orientations in degrees from 0 inclusive to 180
exclusive. The angle describes the stripe normal, so 0° detects
vertical stripes. Each orientation uses the same kernel dimensions,
wavelength, and envelope, with two filter phases 90° apart.

- **Response** is the highest energy among the tested directions.
- **Orientation** reports the winning angle in degrees.
- **Confidence** is the difference between the two strongest
  responses divided by the winning response. A value near zero means
  the responses are close.
- **Valid mask** identifies pixels with a unique winner at or above
  **Minimum response**. Use it to distinguish a valid 0° orientation
  from an invalid pixel. Invalid orientation and confidence values
  are zero.

A mask or region zeros all four outputs outside the selection.
The filter still uses the full image neighborhood. The bank reports
one of the supplied angles, so angular resolution depends on that
list. [OpenCV Gabor kernel reference](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html)

```powershell
opencv-dataflow-run examples/gabor_orientation_bank/gabor_orientation_bank.json
```

[All examples](../README.md)
