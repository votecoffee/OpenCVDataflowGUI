# Tiled inspection mosaic

Open [tiled_scan_mosaic.json](tiled_scan_mosaic.json) and choose **Run all**.
The project batches twelve overlapping views of an original synthetic inspection
strip, registers adjacent views, and displays three windows of the resulting
mosaic. The nominal strip is **18,850 × 144 pixels**. Estimated fractional
translations can add a small uncovered margin.

**Register scan layout** and **Render mosaic tile** are under
**OpenCV → Image Stitching (stitching) → Geometry** in the toolbox. Registration outputs
the source transforms and adjacent-pair fit diagnostics without rendering a
canvas. Each renderer uses that layout to produce one bounded tile. Change
**Tile index** to view another window; indices start at zero and advance across
each row. The example selects indices 0, 18, and 36 using 512-pixel tiles.

- **Image** preserves source pixel depth and channel encoding.
- **Coverage** marks pixels supported by at least one valid source.
  **Overlap** marks pixels supported by at least two.
- **Tile** reports its index, tile count, pixel origin, dimensions, covered and
  overlapping counts, and transforms for the contributing views.
- Tile images and masks carry coordinate metadata mapping local pixels back
  through the mosaic to the first source frame. Image measurements can follow
  this transform; they should not add the tile offset a second time.
- A connected **Tile index** input overrides the parameter, allowing a
  surrounding workflow to request a window without feature refitting.
  Inspect the reported tile count before requesting another index.

Registration accepts **2–64 views**, at most **128 MiB of source pixels** and
8,388,608 pixels per view. Source sides must be below 32,767 pixels. The layout
can describe a mosaic up to 1,000,000 pixels on each side. Rendering allocates
only the selected tile and a feather halo, with a **256 MiB workspace guard**;
reduce Tile size if that guard is reached. Source images and masks remain
resident as a bounded batch. This example renders selected windows; it does
not automatically export every tile.

Choose nearest-neighbor interpolation for exact source samples, or linear for
subpixel registration. Linear windows can differ from a dense render by final
interpolation rounding. Prefer first/last selects source priority. Feather
blending includes a halo so neighboring tile boundaries retain the same edge
weights. Masks select feature centers during registration and valid pixels
during rendering. The tiled renderer supports these priority and feather
modes; exposure correction, graph-cut seams and multiband blending have their
own roadmap scope.

Use ordered, textured views of a mostly planar scene. Registration checks
adjacent feature consensus and optional loop closure. Review fit diagnostics
and coverage before using a layout for inspection. A layout can be serialized
as ordinary JSON and reused with unchanged capture geometry and source grids.
The renderer rejects changed order, labels, sizes, pixel depths or encodings.
Geometric movement of the camera or part requires a fresh registration.

The image strip, project and generator are original synthetic fixtures under
the repository's MIT license. A layout comment repeats this attribution.
Run `python generate.py` from this directory to reproduce the assets and graph.

From the repository root:

```powershell
opencv-dataflow-run "examples/opencv tools/tiled_scan_mosaic/tiled_scan_mosaic.json" --all-images
```

The implementation uses OpenCV's [geometric image transforms](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html).
For small mosaics, see [Stitch scan sequence](../stitch_scan_sequence/README.md).

[All examples](../../README.md)
