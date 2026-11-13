# Inspect structured object collections

Open [object_collections.json](object_collections.json) and run once. This tour
shows that graph connections can carry structured results as well as images.

1. Select **Find all blobs** and inspect its collection, count and measurements.
2. Follow **Keep larger foreground regions** and **Order parts left to right**. Compare their
   output collections with **Ranked parts** to see filtering, stable ordering and
   the resulting image overlay.
3. Inspect the contour and line branches. Each reuses collection operations with
   the relevant measurements; the original detections remain available.
4. Adjust a filter, apply it and rerun. Compare the values on the data output and
   the rendered result. Double-click a result image to examine it more closely.

Next: [choose result graphics](../overlay_result_graphics/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)


## Source and scope

Real ARMBench warehouse photographs (CC BY 4.0), resized and letterboxed.
The fixed-color rule inspects a blue tote cavity. Connected foreground regions
can merge touching items; they are not individual object counts. The two empty
and three loaded labels are example-authored, not publisher occupancy annotations.
The mixed-inventory photo produces four foreground regions; the area filter
retains two. The line branch is illustrative image geometry rather than a part detector.

[Full creator attribution, source hashes and changes](../../assets/showcase/SOURCES.md).
The original synthetic graph remains in [regression fixtures](../../assets/synthetic_graphs/README.md).
