# Follow and edit a small inspection

Open [starter_inspection.json](starter_inspection.json). The bundled scene is
cropped, filtered, thresholded and cleaned before two parts are measured.

1. Run once and select **Measure two parts**. The expected component count is two.
2. Trace the image connections from **Synthetic parts** through **Inspection ROI**,
   **Grayscale**, **Remove noise**, **Segment parts** and **Clean mask**.
3. Select a processing block, change a parameter and use **Run block** to preview
   one execution with its retained input. **Live results** can preview eligible
   edits automatically. Click **Apply configuration** to keep the configuration.
4. Inspect **ROI histogram** and **Part edges** to compare parallel views of the
   same acquisition. Double-click an image in Results for a larger view.

Next: [edit a shared region](../regions/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)
