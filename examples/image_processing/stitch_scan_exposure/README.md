# Correct exposure in an inspection scan

Open [stitch_scan_exposure.json](stitch_scan_exposure.json) with
**File → Open…** and run all three images. The graph stitches the same
three 16-bit tiles twice. Select **Raw mosaic** to see the brightness steps
between tiles. Select **Corrected and feathered mosaic** to see the
exposure-matched output. Use **Layout** to inspect the fit and **Coverage**
to see which pixels are backed by a source view.

The synthetic tiles use 100%, 80%, and 62.5% of the first view's exposure.
Gain correction estimates multipliers of **1.00**, **1.25**, and **1.60**
and reduces the median residual to zero in both overlaps. The corrected
output keeps the input's 16-bit depth and all **49,104 covered pixels**.

- Set **Exposure** to **Gain** for multiplicative brightness differences or
  **Offset** for additive differences. The first view anchors the fit.
  The block samples aligned, valid overlap pixels and excludes very dark and
  saturated intensities. Color fits use the mean of the RGB channels and
  apply the same correction to each color channel. Alpha is left out of the
  correction. Floating-point fits use a nominal full scale of 1.
- Set **Blend** to **Feather** to taper the relative contribution of each view
  near its valid edges, including masked boundaries. **Feather width** is
  measured in mosaic pixels. Coverage and overlap still describe the actual
  source support.
- When a pair has too few usable samples or would exceed the configured
  correction limits, **Layout** reports that pair and the block withholds
  the mosaic. Exposure correction and feathering both change measured pixel
  values. Use **None** and **Prefer first** to retain the default behavior.

```powershell
opencv-dataflow-run examples/image_processing/stitch_scan_exposure/stitch_scan_exposure.json --all-images
```

[All examples](../../README.md)
