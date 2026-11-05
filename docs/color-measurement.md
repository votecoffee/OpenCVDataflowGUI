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

Open the [Lab color match example](../examples/lab_color_match.json), or run:

```powershell
opencv-dataflow-run examples/lab_color_match.json
```

Both sample regions in the example measure the same synthetic patch and
produce ΔE = 0. The graph compares mean colors. Use the per-channel
standard deviations when judging how uniform each selected area is.
Sample and reference Lab values should use consistent illumination,
camera settings and reference-white assumptions. Camera color-chart
calibration remains a separate step.
