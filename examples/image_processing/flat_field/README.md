# Flat-field illumination correction

Open [flat_field.json](flat_field.json) with **File → Open…**, then click
**Run once**. The bundled grayscale sample has a left-to-right lighting gradient
and a darker rectangular defect. **Flat-field correction** uses the bright and
dark reference images to flatten the background while preserving the defect.
The sample background spans 59 gray levels before correction and is constant
afterward. All 9,600 pixels are corrected in this example.

For your own images, capture a uniformly illuminated flat reference and an
optional dark reference with the same camera, optics, exposure, gain, and pixel
grid. The input and references must have the same dimensions, channel encoding,
and depth. Correction computes `(image − dark) / (flat − dark) × target` in each
intensity channel. Without a dark reference, dark is zero. [OpenCV array division](https://docs.opencv.org/5.0/javadoc/org/opencv/core/Core.html)

- **Target mode** can preserve each channel's mean usable flat response or set
  all channels to a fixed level. **Target level** and **Min response** use
  fractions of full scale for integers and sample units for float32.
- The block preserves uint8, uint16, or float32 depth, the channel order, and
  alpha. Integer outputs round and saturate. A connected mask or region limits
  correction; the reference mean still uses the full usable reference.
- Pixels whose flat-minus-dark response is too small in any intensity channel
  remain unchanged and appear in **Invalid mask** if selected. The block reports
  a failed result if a nonempty selection has no usable pixels.

```powershell
opencv-dataflow-run examples/image_processing/flat_field/flat_field.json
```

[All examples](../../README.md)
