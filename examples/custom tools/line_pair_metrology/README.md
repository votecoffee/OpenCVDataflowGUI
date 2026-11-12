# Line-pair metrology

Open [corner.json](corner.json) or [gap.json](gap.json) and choose **Run all**.
Each project loads two original synthetic fixtures: a passing image and a
failing image. Separate crops isolate the two edges, Hough detection finds
line segments, sorting selects the longest, and Fit shape fits each line.
**Measure line pair** maps both crop frames into the original image before
measuring. Its result image shows line support and a separate measurement
overlay layer.

The corner project expects **90 ± 2 degrees**. Its second fixture is about
80 degrees and fails the angle check. Both corners lie near original-image
pixel (60, 60), outside the fitting crops. The gap project expects **19–21
pixels** of separation; the fixtures have 20- and 30-pixel gaps.

Find the block under **OpenCV → Computational Geometry (geometry) → Measurements**.

- **Corner** reports the intersection and the selected acute or obtuse angle.
  Nearly parallel lines, excessive extrapolation and an out-of-image vertex
  can make the intersection unmeasurable.
- **Angle** measures orientation without requiring an intersection or shared
  support. Parallel lines have a valid zero-degree acute angle.
- **Parallel gap** requires overlapping projected segment support and an angle
  within Maximum parallel angle. Gap is the transverse separation at the
  overlap midpoint, measured normal to line A. The report also gives minimum
  and maximum separation across the overlap; gap limits apply to that full
  range. This distinguishes a taper from a constant-width gap.
- Line inputs accept `inspect.fit_shape` line records or a line collection
  with a configurable index. Plain unframed records use current image pixels.
  Known crop/pyramid frames must share the image's root; mismatched tickets
  are rejected.
- For physical measurements, connect **Rectify workplane** Scale and Valid mask,
  and select the matching units. Lengths use its uniform units-per-pixel scale;
  vertex coordinates also use its physical origin. Use Vertex pixels for image-space
  consumers; physical Vertex data carries its workplane coordinate descriptor.
- Optional Valid mask and Region inputs require all measured support paths to
  remain covered, including a pixel of surrounding support. Physical mode
  requires the validity mask.
- A missing selected detection is a failed measurement. Malformed lines or
  incompatible frames/tickets are invalid inputs. Unmeasurable geometry
  withholds numeric results; out-of-tolerance geometry retains its measured
  values and emits a specific fail reason.

Pixels and their encoding pass through unchanged. Result graphics are drawable
objects and honor the block/global graphics settings. Choose which layers to
display or aggregate elsewhere in the program.

The images, generator and projects are original synthetic fixtures under the
repository's MIT license. Both layouts include an attribution comment.
Run `python generate.py` here to recreate them.

From the repository root:

```powershell
opencv-dataflow-run "examples/custom tools/line_pair_metrology/corner.json" --all-images
opencv-dataflow-run "examples/custom tools/line_pair_metrology/gap.json" --all-images
```

Each command deliberately reports one pass and one inspection failure and
returns a nonzero status for the failing fixture.

[All examples](../../README.md) · [Block parameters](../../../docs/block-parameters.md)
