# Run-linked contours

**Run-linked contours** (`inspect.run_contours`) is under
**OpenCV → Image Processing → Shapes**. It uses `findContoursLinkRuns` for large
sparse uint8 monochrome masks. Every nonzero value is foreground. Color, boolean,
high-depth, physical-depth and empty arrays require explicit upstream conversion
or are rejected. The existing Find contours and Contour hierarchy tools provide
the ordinary boundary-following and nested-tree workflows.

## Boundary and hierarchy contract

The qualified OpenCV 5.0.0 wheel returns flat hierarchy links for this API,
despite the native documentation describing two-level CCOMP hierarchy.
This block deliberately offers **List** retrieval and the native **Simple**
mode only. It publishes consistent flat sibling links with no child/parent IDs;
it does not infer hole ownership or reconstruct nested topology.

Point order, winding and polygon geometry differ from ordinary Simple contours.
Vertical run endpoints may remain collinear. Solid rectangle areas, perimeters,
bounds and rasterizations agree with the ordinary reference in the supplied
fixtures. Hole polygons can differ: the 23×19 nested fixture has run polygon
areas **4, 48, 196**, while ordinary contours give **4, 62, 196**. Its true tree
has depths 0, 1 and 2. Neither area list represents foreground pixel counts.

**Data** explicitly reports `boundary_model=native_run_endpoints`,
`hierarchy_semantics=flat_list`, `hole_parents_available=false`, horizontal run
count and raw contour/point counts. Measurements and collection filters use
these native polygons. Use ordinary contours when that boundary convention or
hole/tree relationships are required.

## Controls, coordinates and limits

- **Min area** filters native polygon areas. **Expected count** judges the
  retained contour count; a mismatch emits a valid failed inspection and reason.
- Optional masks and regions intersect. **Center** keeps whole polygons whose
  centers are selected; **Clip** changes the search pixels and resulting geometry.
  Known coordinate frames must agree.
- **Offset** shifts contour coordinates and the collection/data frame while
  raster outputs and drawable graphics stay on the input grid. The full shifted
  grid must fit signed 32-bit coordinates. Tickets, acquisition metadata and
  pyramid mappings follow the normal payload contract.
- **Max runs** preflights actual horizontal runs in the search image before the
  native call. It scans strips of at most 65,536 pixels instead of creating a
  second full binary image. Complex checkerboard-like input can be rejected
  even if area filtering would later discard its contours.
- **Max contours** and **Max points** reject oversized native output before
  filtering, JSON measurements and graphics. They do not silently truncate.
  Native output allocation occurs before those two checks; the run preflight
  limits its input complexity, not every internal allocator byte.
- **Draw overlay** and **Draw mask** enable optional full-size raster outputs;
  both default off. The mask is a rasterization of retained native polygons and
  need not reconstruct the source pixels. Drawable graphics are separate and
  respect block/global graphics settings.

Inputs are bounded to 16,384 per axis and 64 million pixels. Default limits are
250,000 runs, 10,000 contours and 250,000 native vertices. Raising these limits
can make native buffers, Python collections, payload snapshots and graphics large.
These checks do not isolate native processing in another process.

## Measured allocation and timing

The reproducible [native benchmark](<../examples/opencv tools/run_contours/benchmark_native.py>)
uses a fresh Windows process per API/size, one OpenCV thread, 16 known solid
rectangles and three timed native calls after a small warmup. Both workers also
measure the bounded run preflight separately. Observed peak commit increases:

| Square mask | Ordinary List/Simple | Link runs | Native median, ordinary / runs |
| --- | ---: | ---: | ---: |
| 2048² | 4.12 MiB | 0.00 MiB additional peak | 1.31 / 0.27 ms |
| 6000² | 34.44 MiB | 0.62 MiB | 13.95 / 2.49 ms |
| 8000² | 61.21 MiB | 0.72 MiB | 25.73 / 3.33 ms |

At 8000² the run preflight took 16.26 ms in the run worker. These are observations
on this build and fixture, not whole-block or graph performance guarantees.
Zero additional process peak does not mean zero allocation. Graph snapshots,
measurement construction, JSON and raster outputs are excluded from native timing.
Run-linked contours retained 12,800 vertices at 8000² versus the ordinary API's
64, so a sparse input or smaller native temporary allocation does not guarantee
a smaller final coordinate collection.

[Five original projects](<../examples/opencv tools/run_contours/README.md>) ·
[OpenCV native API](https://docs.opencv.org/5.0/main_modules/imgproc_shape.html) ·
[Native implementation](https://github.com/opencv/opencv/blob/5.x/modules/imgproc/src/contours_link.cpp)
