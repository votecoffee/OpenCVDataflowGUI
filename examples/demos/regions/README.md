# Editable inspection regions

Learn how a region travels through the graph and limits work on an image.
Open [regions.json](regions.json), then run once.

1. Select **Circle selects left PCB area** and open its region editor in the inspector.
   Resize or move the circle, apply the change, and run again.
2. Compare **Measure left PCB area** with **Blur only inside circle**. They share the
   region, while the original source image remains available to other branches.
3. Select **Rectangle selects right PCB area** and **Polygon selects the board** to try
   different region shapes. Inspect **Histogram inside polygon** and its values.

The purpose is to learn the region editor and typed connections. Advanced region
composition and transforms are in the Custom tools collection.
Next: [inspect object collections](../object_collections/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)


## Source and scope

Real VisA PCB photograph (CC BY 4.0), resized and letterboxed to 480×320.
Thresholded components illustrate the workflow; their count is not a defect label.

[Full creator attribution, source hashes and changes](../../assets/showcase/SOURCES.md).
The original synthetic graph remains in [regression fixtures](../../assets/synthetic_graphs/README.md).
