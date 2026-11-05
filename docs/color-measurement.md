# Lab color measurement

**Measure Lab color** measures the selected image area. Connect a mask or
region, or leave both disconnected to measure the whole image. The block
outputs mean CIELAB coordinates, per-channel standard deviations, the
number of measured pixels and a preview swatch. It uses OpenCV's color
conversion for normalized color channels and converts OpenCV's uint8
Lab encoding to conventional L*, a*, b* values.
See the [OpenCV color conversions reference](https://docs.opencv.org/5.0/main_modules/imgproc_color_conversions.html).

**Compare Lab colors** accepts a measured sample and a second measured
reference or a configured reference value. Choose CIE76 or CIEDE2000 and
set the maximum difference that passes. The block reports ΔE and signed
sample-minus-reference channel differences. CIEDE2000 uses configurable
lightness, chroma and hue divisors.
See the [CIEDE2000 reference data and implementation notes](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/).

Open the [Lab color match example](../examples/lab_color_match/lab_color_match.json), or run:

```powershell
opencv-dataflow-run examples/lab_color_match/lab_color_match.json
```

Both sample regions in the example measure the same synthetic patch and
produce ΔE = 0. The graph compares mean colors. Use the per-channel
standard deviations when judging how uniform each selected area is.
Sample and reference Lab values should use consistent illumination,
camera settings and reference-white assumptions.


## Fit and apply a color-chart correction

Open the [color-chart calibration example](../examples/color_chart_calibration/color_chart_calibration.json)
or run `opencv-dataflow-run examples/color_chart_calibration/color_chart_calibration.json`.

1. **Detect color chart** finds a Macbeth, Digital SG or Vinyl chart and
   emits ordered RGB patches, chart geometry and detection cost. The
   optional region limits the search area and selects charts by patch
   centers. Keep the full chart inside the selected area.
2. **Fit color correction model** accepts the detected chart or ordered
   RGB patch triplets in [0, 1]. Use the corresponding built-in reference
   or supply custom sRGB or linear-sRGB values in the same patch order.
   It reports the 3×3 or 4×3 matrix, native fit loss and retained patch
   count. Select identity, gamma or per-channel polynomial linearization.
   Weighted fitting is available for 3×3 models in this build.
3. **Apply color correction model** accepts the fitted model over a wire
   or as JSON in the block configuration. The JSON can also be stored in
   a recipe field bound to the block's model parameter. Apply it to images
   from the same camera setup and lighting. The block keeps the image
   layout, depth and alpha, and can restrict correction to a mask or region.

The model outputs sRGB and retains the fitted matrix and linearization.
Automated checks use a synthetic Macbeth chart and held-out RGB samples.
Digital SG and Vinyl are selectable. Those chart types and physical-camera
calibration still need acceptance with representative images.
See the [OpenCV chart detector](https://docs.opencv.org/5.0/tutorials/objdetect/macbeth_chart_detection/macbeth_chart_detection.html)
and [color correction model](https://docs.opencv.org/5.0/tutorials/photo/ccm/color_correction_model.html).
