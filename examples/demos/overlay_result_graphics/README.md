# Compose and choose result graphics

Open [overlay_result_graphics.json](overlay_result_graphics.json) and run once
in **Debug**. The region outline and contour annotations are drawable objects;
the contour measurements come from a separate thresholded branch.

1. Select **Overlay result graphics** and inspect **Result / image**.
2. Open **Choose graphics…**, uncheck a region or contour layer, click
   **Apply configuration**, and run again.
3. Compare the original color image with the chosen overlay. Double-click the
   result image to open a larger view.

This illustrates independent processing/graphics connections and selecting which
measurements appear on a final image. Optional graphics can be disabled by a run
profile, so use Debug for this tour.
Next: [run an image set](../image_set_workspace/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)
