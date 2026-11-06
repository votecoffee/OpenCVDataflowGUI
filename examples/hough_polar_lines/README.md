# Hough polar lines

Open [hough_polar_lines.json](hough_polar_lines.json) with **File → Open…**
and click **Run once**. Both branches get the same synthetic edge map.
Binary voting selects a horizontal line with 108 edge pixels. Weighted voting
selects the shorter, brighter vertical line with an intensity sum of 10,560.
The two **Draw object collection** blocks display their selected line spans.

Find **Hough polar lines** under **Inspection → Features**. Supply an 8-bit
monochrome edge map. Binary mode counts each nonzero pixel once. Weighted mode
uses the pixel value, so a binary Canny image gives equal weight to every edge
pixel. A mask or region restricts which pixels vote. **Minimum votes** is
inclusive and uses accumulator units for the selected mode.

**Polar lines** reports signed rho in pixels, normal angle in degrees, the
native accumulator value, and the number of pixels supporting each span. A
normal near 0° describes a vertical line. A normal near 90° describes a
horizontal line. **Support distance** controls which edge pixels determine the
endpoints. Zero uses half the rho bin spacing. Peaks without nearby pixels
appear only in the polar report. Use **Strength** in collection tools to
filter or sort by accumulator value.

The block uses OpenCV's standard Hough transform. The maximum line count caps
the report after OpenCV ranks the peaks. Use a sparse edge map with up to one
million voting pixels and coarser bin sizes or a narrower angle range for
larger images. [OpenCV Hough line reference](https://docs.opencv.org/5.0/main_modules/imgproc_feature.html#houghlines)

```powershell
opencv-dataflow-run examples/hough_polar_lines/hough_polar_lines.json
```

[All examples](../README.md)
