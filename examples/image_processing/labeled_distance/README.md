# Labeled distance transform

Open [labeled_distance.json](labeled_distance.json) with **File → Open…**
and click **Run once**. Two circular regions form a seed mask.
**Labeled distance transform** assigns each pixel to its nearest seed
region. **Labels** shows the two zones and **Selected mask** isolates
the zone for label 1. **Distance** reports distance to the nearest seed
pixel in source-image pixels.

Find **Labeled distance transform** under **Image processing → Morphology**.
Wire a monochrome uint8 seed image and choose zero or nonzero pixels as
seeds. Optional mask and region inputs restrict eligible seeds while the
distance and label outputs cover the full input image. **Components**
labels each 8-connected seed region. **Pixels** gives every seed pixel
its own label. [OpenCV distance transform reference](https://docs.opencv.org/5.0/main_modules/imgproc_misc.html#distancetransform)

Choose exact L1 or chessboard distance, or OpenCV's approximate L2
distance with a 5 × 5 neighborhood. The distances are float32 and the
labels are int32. Ties follow OpenCV's assignment. **Selected label**
uses the label values from the current run. Set it to zero to disable
the mask. The input limit is 16,384 pixels per axis and 16,777,216 total.

```powershell
opencv-dataflow-run examples/image_processing/labeled_distance/labeled_distance.json
```

[All examples](../../README.md)
