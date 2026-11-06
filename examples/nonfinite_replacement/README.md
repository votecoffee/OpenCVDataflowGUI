# Replace nonfinite samples

Open [nonfinite_replacement.json](nonfinite_replacement.json) with
**File → Open…** and click **Run once**. The first check finds a 100-pixel
NaN patch in a synthetic depth image and sends its invalid mask to
**Replace nonfinite samples**. The repair uses a configured value of
3 meters, the known depth of this synthetic plane. The final check
finds zero invalid pixels in the 2.9 to 3.1 meter range.

Find **Replace nonfinite samples** under **Image processing → Color and depth**.
It accepts float32 and float64 grayscale, color, and physical depth images.
Use **Mode** to replace all nonfinite samples, only NaN, or only infinity.
NaN, positive infinity, and negative infinity each have their own finite
replacement value in the source's stored units.

Optional mask and region inputs restrict changes to their intersection.
Each selected channel is checked independently. Finite samples and values
outside the selection are preserved. The output retains the source dtype,
color encoding, and physical depth units. **Replaced mask** marks each
pixel with at least one replacement. **Report** counts replaced channel
samples by kind, so the sample count can exceed the pixel count.

The filled depth values come from the block configuration. The replacement
mask identifies them for downstream tools. [OpenCV NaN replacement reference](https://docs.opencv.org/5.0/main_modules/core_array.html#patchnans)

```powershell
opencv-dataflow-run examples/nonfinite_replacement/nonfinite_replacement.json
```

[All examples](../README.md)
