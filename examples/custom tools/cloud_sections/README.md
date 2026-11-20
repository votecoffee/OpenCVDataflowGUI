# Measure a channel at two section heights

Open [cloud_sections.json](cloud_sections.json) and run the graph. The PLY
stores a rotated millimeter scan. The fixture pose maps section coordinates
into the scan frame. Two blocks cut 2.5 mm thick XY slabs at 25 mm and 55 mm
above the fixture origin.

| Section | Observed widths by U bin | Coverage |
| --- | --- | ---: |
| Lower, 25 mm | 16, 18, gap, 20, 24 mm | 80% |
| Upper, 55 mm | 28, 28, 32, 32, 36 mm | 100% |

The U interval is 0–125 mm with five equal bins. Both sections meet the
75% minimum coverage. The lower section contains eight points and the upper
contains ten. Additional points outside the U/V window or the slab are excluded.

Inspect **Profile** for the minimum, mean, and maximum V in meters, the
width, and the original input rows for each measured bin. Each width is
the span between the two observed extremes. A bin with fewer than two
points in this project reports no width. The plot leaves a gap there.
Interior bin boundaries belong to the bin on their right. The last bin
includes the end of the configured U interval.

**Report** includes distance to the section plane and the observed U/V
extent. **Section** contains the original selected points and attributes
in the scan frame and in input order. The measured envelope uses only
observed points, so the configured coverage helps detect missing support.

The [builder](build_example.py) and fixture are original OpenCVDataflowGUI
contributions under the repository's MIT license.
