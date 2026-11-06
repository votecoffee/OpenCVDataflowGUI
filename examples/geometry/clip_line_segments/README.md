# Clip line segments

Open [clip_line_segments.json](clip_line_segments.json) with **File → Open…**
and click **Run once**. Hough detection finds two edges in a synthetic image.
**Shape region** sets a rectangle around part of the horizontal edge.
**Clip line segments** retains a 51-pixel span of that edge and rejects the
vertical edge outside the window. The last block draws the retained span and
produces its mask.

Find **Clip line segments** under **Inspection → Collections**. Wire a line
collection and, optionally, a non-inverted rectangle from **Shape region**.
The rectangle must use the image frame that produced the lines. The block
clips against its first and last pixel centers, using a 1/1024-pixel grid
when computing new endpoints. It preserves original endpoints that are
already inside the window. With no region wire, the image bounds are used.
[OpenCV clipping reference](https://docs.opencv.org/5.0/main_modules/imgproc_draw.html#clipline)

Set **Minimum length** to exclude short spans or point contacts. The
`collection` output can feed **Sort object collection** or **Draw object
collection**. `rejected` contains the original objects for any spans that
missed the window or fell below the minimum.

```powershell
opencv-dataflow-run examples/geometry/clip_line_segments/clip_line_segments.json
```

[All examples](../../README.md)
