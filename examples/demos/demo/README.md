# Quick-start graph

Learn the basic run/select/results cycle on a real VisA PCB photograph. Open [demo.json](demo.json) with **File → Open…**.

1. Press **Run once** (F5). Select **Find contours** and inspect its result image
   and count; the expected large foreground contour count is one.
2. Select **Grayscale**, **Gaussian blur** and **Threshold** to follow the images
   along their wires. Hover connectors to see their full names and simple values.
3. Move a node, choose **Workspace → Auto arrange blocks**, then save the project.
   Reopen it to retain the saved layout.

Next: [build a small inspection](../starter_inspection/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)


## Source and scope

Real VisA PCB photograph (CC BY 4.0), resized and letterboxed to 480×320.
Thresholded components illustrate the workflow; their count is not a defect label.

[Full creator attribution, source hashes and changes](../../assets/showcase/SOURCES.md).
The original synthetic graph remains in [regression fixtures](../../assets/synthetic_graphs/README.md).
