# Image sets, tickets and a grouped workspace

Open [image_set_workspace.json](image_set_workspace.json). This tour uses three
real ARMBench warehouse captures and a saved grouped layout.

1. Click **Run all** to inspect the set once. **Continuous** repeats the set until
   stopped. Each filename becomes the acquisition's ticket.
2. Inspect passing and failing thumbnails in the result filmstrip. Double-click
   a thumbnail to inspect that retained result in a larger view.
3. Select **Final ticket judgment** to see how a processing result becomes a
   ticket verdict. Failed inspections are intentional parts of this demo.
4. Move a group, pan or zoom using the workspace/minimap, then save and reopen.
   The graph keeps its saved layout and view.

Next: [score a validation set](../validation_images/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)


## Source and scope

Real ARMBench warehouse photographs (CC BY 4.0), resized and letterboxed.
The fixed-color rule inspects a blue tote cavity. Connected foreground regions
can merge touching items; they are not individual object counts. The two empty
and three loaded labels are example-authored, not publisher occupancy annotations.

[Full creator attribution, source hashes and changes](../../assets/showcase/SOURCES.md).
The original synthetic graph remains in [regression fixtures](../../assets/synthetic_graphs/README.md).
