# Follow and edit a small inspection

Open [starter_inspection.json](starter_inspection.json). The real VisA PCB photo is
cropped, filtered, thresholded and cleaned before its large foreground is measured.

1. Run once and select **Measure the PCB foreground**. The expected selected component count is one.
2. Trace the image connections from **Real VisA PCB photograph** through **Inspection ROI**,
   **Grayscale**, **Remove noise**, **Segment parts** and **Clean mask**.
3. Select a processing block, change a parameter and use **Run block** to preview
   one execution with its retained input. **Live results** can preview eligible
   edits automatically. Click **Apply configuration** to keep the configuration.
4. Inspect **ROI histogram** and **Part edges** to compare parallel views of the
   same acquisition. Double-click an image in Results for a larger view.

Next: [edit a shared region](../regions/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)


## Source and scope

Real VisA PCB photograph (CC BY 4.0), resized and letterboxed to 480×320.
Thresholded components illustrate the workflow; their count is not a defect label.

[Full creator attribution, source hashes and changes](../../assets/showcase/SOURCES.md).
The original synthetic graph remains in [regression fixtures](../../assets/synthetic_graphs/README.md).
