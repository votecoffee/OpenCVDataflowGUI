# Editable inspection regions

Learn how a region travels through the graph and limits work on an image.
Open [regions.json](regions.json), then run once.

1. Select **Circle selects left part** and open its region editor in the inspector.
   Resize or move the circle, apply the change, and run again.
2. Compare **Measure left part** with **Blur only inside circle**. They share the
   region, while the original source image remains available to other branches.
3. Select **Rectangle selects right part** and **Polygon excludes text** to try
   different region shapes. Inspect **Histogram inside polygon** and its values.

The purpose is to learn the region editor and typed connections. Advanced region
composition and transforms are in the Custom tools collection.
Next: [inspect object collections](../object_collections/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)
