# Stitch an ordered inspection scan

Open [stitch_scan_sequence.json](stitch_scan_sequence.json) with
**File → Open…** and run all three images. **Batch serial entries** collects
the overlapping views in file order. **Stitch scan sequence** matches SIFT
features between each adjacent pair, estimates their translation, and warps
each view once into the mosaic. The synthetic tiles come from the same
372 × 132 field. The output has **49,104 covered pixels** and **11,088**
overlapping pixels per adjacent pair. Small estimated subpixel offsets can
leave uncovered margin rows in the canvas.

Find **Stitch scan sequence** under **Image processing → Geometry**.

- Supply an ordered batch of two to eight views with matching pixel depth
  and encoding. The first view defines the mosaic's orientation and scale.
  Choose **Similarity** when neighboring views also rotate or change scale.
- If needed, batch one aligned uint8 mask per view in the same order.
  The masks select feature centers and limit the rendered coverage.
- **Coverage** marks every pixel supported by a view. **Overlap** marks
  pixels supported by at least two. **Layout** reports source mappings,
  view tickets, adjacent inliers, reprojection error, and overlap counts.
  The mosaic carries the Batch ticket.
- If a join falls below the fit or coverage threshold, Layout identifies the
  failed pair and the block withholds the mosaic. **Prefer first** keeps
  earlier covered pixels. **Prefer last** gives priority to later views.

Use ordered, textured views of a mostly planar scene. The registration
follows adjacent pairs, so errors can accumulate along a long scan.
The example uses nearest-neighbor resampling to verify the covered pixels
against their source tiles.
[OpenCV feature matching](https://docs.opencv.org/5.0/py_tutorials/py_features/py_matcher/py_matcher.html)

```powershell
opencv-dataflow-run examples/image_processing/stitch_scan_sequence/stitch_scan_sequence.json --all-images
```

[All examples](../../README.md)
