# Contour defect inspection

Open [contour_defects.json](contour_defects.json) with **File → Open…**.
Click **Run once** for each of the two images. **Find contours** extracts
the part boundary, and **Contour hull and convexity defects** checks for
concavities at least 8 pixels deep.

- **shallow_notch.png** passes. The shallow notch is 4 pixels deep and
  the qualifying count is zero.
- **deep_notch.png** fails with `too_many_convexity_defects`. Its deepest
  qualifying notch is 18 pixels deep.

The final overlay draws the hull in green, depths below the cutoff in
amber, and qualifying depths in red. Each is a separate result graphics
layer. **Maximum depth** reports the deepest qualifying concavity, or
zero when none qualify. To tolerate one qualifying concavity, set
**Maximum defects** to 1. Turn off **Check defect count** to measure
without assigning a verdict based on count.

OpenCV reports the deepest point between successive hull vertices.
Multiple notches along a single hull edge can therefore contribute one
measurement. The tool measures concavities of a single outer contour
against its own hull. It cannot count internal holes or detect a missing
corner that the hull follows. Use a simple whole-pixel contour for defect
measurements. [OpenCV shape analysis reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html)

```powershell
opencv-dataflow-run examples/contour_defects/contour_defects.json --all-images
```

The command reports one pass and one failure and returns exit code 1.

[All examples](../README.md)
