# Skeletonization and spur pruning

From the repository root, install the optional morphology dependency.

```powershell
python -m pip install -e ".[morphology]"
```

Open [skeleton_pruning.json](skeleton_pruning.json) and click **Run once**.
The bundled binary image has a thick horizontal trace with a short branch.
**Skeletonize mask (scikit-image)** finds the centerline, then **Prune skeleton
spurs** removes the short branch. Select the pruning block to inspect its
cleaned mask, removed-pixel mask, and preview. Retained pixels appear green
and removed pixels red.

Skeletonize accepts a two-dimensional uint8 or boolean mask. Select Zhang,
Lee, Guo–Hall, or medial axis in **Method**. Medial axis also emits a
floating-point distance-to-background image. For a grayscale photograph,
threshold it first so that foreground pixels are nonzero.
[Scikit-image morphology reference](https://scikit-image.org/docs/stable/api/skimage.morphology.html)

Pruning measures path length from an endpoint to a junction, counting cardinal
steps as one pixel and diagonal steps as the square root of two pixels. It
keeps junction pixels, loops, and standalone lines. A connected region or mask
limits removal to spurs whose removable pixels all fall within the selection.
Active pruning accepts up to 250,000 foreground skeleton pixels.
[Scikit-image pixel graph reference](https://scikit-image.org/docs/stable/api/skimage.graph.html#skimage.graph.pixel_graph)

```powershell
opencv-dataflow-run examples/image_processing/skeleton_pruning/skeleton_pruning.json
```

[All examples](../../README.md)
