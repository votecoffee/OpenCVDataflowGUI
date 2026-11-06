# Check image samples

Open [image_validity.json](image_validity.json) with **File → Open…** and
click **Run once**. The synthetic source produces a 64 × 64 depth image
with a missing 10 × 10 patch in the upper left. **Check image samples**
finds 100 nonfinite pixels, or 2.44% of the frame, which passes the example's
3% limit. The valid mask limits the intensity statistics to 3,996 pixels
that have valid depth in the same image frame.

Find **Check image samples** under **Inspection → Measurements**. It takes
grayscale, color, and physical depth images. Every stored channel must be
finite. Enable **Check range** to also check every channel against the
inclusive minimum and exclusive maximum in the source's units.

Optional mask and region inputs restrict the check to their intersection.
Both output masks are zero outside the selection. A pixel with a
nonfinite channel counts once as nonfinite. The out-of-range count covers
only pixels whose channels are all finite. The report identifies the first
invalid pixel in row-major order, using local `[x, y]` coordinates.
The block emits masks and a report even when the check fails. Downstream
blocks receive the pass or fail status with those outputs.

Ordinary floating-point image previews show nonfinite pixels in magenta.
Physical depth keeps its depth-specific preview. [OpenCV finite mask reference](https://docs.opencv.org/5.0/main_modules/core_array.html#finitemask)

```powershell
opencv-dataflow-run examples/image_processing/image_validity/image_validity.json
```

[All examples](../../README.md)
