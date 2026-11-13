# Contour rasterization with holes

Open [project.json](project.json) and select **Run once**. The original binary
cutout fixture contains an outer part, a hole, an island with another hole,
and an independent solid part. **Contour hierarchy** sends the full collection
to three **Rasterize contours with holes** blocks:

| Branch | Result | Foreground pixels |
| --- | --- | ---: |
| Preserve all topology | Both parts, including the nested hole and island | 15,168 |
| Select one part with holes | The larger part and all its descendants | 12,032 |
| Explicitly fill all holes | Union of every filled boundary | 21,568 |

Each branch checks its pixel count and compares its mask with a separate
slice-based reference. Every maximum absolute pixel error must be zero.
The area references use rectangle arithmetic and do not call the rasterizer.
The optional image input supplies a matching source image for overlays and an
unchanged image output. Mask samples are always uint8 zero/255.

Use the full hierarchy and select **Indices** or **IDs** inside the rasterizer
to keep a selected part's child holes. Indices follow the collection's current
order; IDs survive sorting but are local to that detection run. This fixed
fixture's selected root ID is recorded in [truth.json](assets/truth.json).
**Include descendants** and **Maximum depth** control nested boundaries.
A selected hole becomes a new foreground root with relative descendant parity.

**Hierarchy** groups each related subtree before unioning independent roots.
It rejects missing, cyclic or inconsistent links, including links lost by
filtering a tree before rasterization. **Solid** explicitly ignores topology
and unions all selected polygons. **Even odd** explicitly applies parity to
selected contour outlines together, so independent overlapping contours may
cancel their shared interiors. Blob items are independent groups in every mode.
These distinctions follow the [OpenCV drawing reference](https://docs.opencv.org/5.0/main_modules/imgproc_draw.html).

Filled masks use `drawContours` at whole-pixel precision and `fillPoly` with
fixed-point precision. Outlines use `polylines` with configurable thickness.
Choose 4- or 8-connected binary boundaries; masks do not contain antialiasing
coverage. Subpixel coordinates are rounded to the chosen fixed-point precision.
Mask/region clipping occurs after rasterization and optional inversion; pixels
outside the selection remain zero. Contours outside the canvas are clipped.

The collection's origin is removed to recover its image grid. A reference
image must have matching dimensions and compatible known frames and tickets.
**Pixel grid** explicitly permits different equal-sized frames; it does not
override ticket checks. Output payloads preserve spatial, pyramid and companion
metadata. Supplied image pixels, color encoding and depth are retained.
A known collection frame remains available when the reference image has no
named coordinate frame.

Limits: 64M canvas pixels, 16,384 pixels per axis, 100,000 vertices per boundary,
one million selected vertices, and 256M summed clipped group-bounding-box
pixels. Diagnostic outlines are limited to 500 boundaries and 20,000 vertices;
the raw mask uses all qualified geometry. Disable result graphics to skip
overlay creation. This is raster topology verification, not physical-area or
manufacturing acceptance qualification.

```powershell
opencv-dataflow-run "examples/opencv tools/contour_rasterization/project.json"
```

Regenerate the fixture with `python "examples/opencv tools/contour_rasterization/generate.py"`.
All images, code and references are original material under the repository's
MIT license. The project includes an attribution/comment block.

[All examples](../../README.md)
