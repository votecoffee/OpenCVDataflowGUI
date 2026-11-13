# Flat-field illumination correction

Open [flat_field.json](flat_field.json) with **File → Open…**, then click
**Run once**. The bundled grayscale sample has a left-to-right lighting gradient
and a darker rectangular defect. **Flat-field correction** uses the bright and
dark reference images to flatten the background while preserving the defect.
The sample background spans 59 gray levels before correction and is constant
afterward. All 9,600 pixels are corrected in this example.

For your own images, capture a uniformly illuminated flat reference and an
optional dark reference with the same camera, optics, exposure, gain, and pixel
grid. References normally match the input depth and channel order; RGBA/BGRA
images can also use RGB/BGR references. Correction computes `(image − dark) / (flat − dark) × target` in each
intensity channel. Without a dark reference, dark is zero. [OpenCV array division](https://docs.opencv.org/5.0/javadoc/org/opencv/core/Core.html)

- **Target mode** can preserve each channel's mean usable flat response or set
  all channels to a fixed level. **Target level** and **Min response** use
  fractions of full scale for integers and sample units for floating images.
- The block preserves uint8/uint16/float32/float64 depth, the channel order, and
  alpha. Integer outputs round and saturate. A connected mask or region limits
  correction; the reference mean still uses the full usable reference.
- Pixels whose flat-minus-dark response is too small in any intensity channel
  remain unchanged and appear in **Invalid mask** if selected. The block reports
  a failed result if a nonempty selection has no usable pixels.

[Build references from capture batches](../flat_field_reference/README.md) to
reduce capture noise with mean or median estimates. For floating references in
original integer counts, select **Reference depth > Raw sample units**; this
performs no full-scale conversion. **Reference alignment > Same frame** can
check known camera grids; the default compares already aligned pixel indices.
Use straight alpha, unassociating premultiplied images before correction.

```powershell
opencv-dataflow-run "examples/custom tools/flat_field/flat_field.json"
```

[All examples](../../README.md)
