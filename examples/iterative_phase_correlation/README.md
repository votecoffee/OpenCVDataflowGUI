# Iterative phase correlation

Open [iterative_phase_correlation.json](iterative_phase_correlation.json)
with **File → Open…** and click **Run once**. The graph moves a synthetic
inspection image right by 4.3125 pixels and up by 5.6875 pixels. The
correlation block estimates that shift and sends the inverse translation
to **Warp affine**. Inspect **Shift** for the measured displacement and the
final warp for the realigned image. Resampling makes the realignment
approximate.

Find **Iterative phase correlation** under **Inspection → Registration**.
Connect the reference to **Image A** and the moving image to **Image B**.
Use matching even widths and heights of at least 4 pixels, up to
8,388,608 total pixels per image. Crop matching regions in both images
when inspecting a local feature. Color images are measured in grayscale.
The OpenCV routine applies a Hann window. **Neighborhood size** and
**Max iterations** control its subpixel refinement.
[OpenCV iterative phase correlation reference](https://docs.opencv.org/5.0/main_modules/imgproc_motion.html#phasecorrelateiterative)

**Shift** reports how Image B is displaced from Image A in local pixels.
**Alignment transform** moves B toward A. The block reports a failed
status and withholds these outputs when an image interior is flat in
the selected precision, the native shift is nonfinite, or the shift
reaches the half-image wraparound boundary. The estimate assumes
translation and depends on having distinctive shared texture.

```powershell
opencv-dataflow-run examples/iterative_phase_correlation/iterative_phase_correlation.json
```

[All examples](../README.md)
