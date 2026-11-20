# Measure height and spread against a datum

Open [cloud_datum.json](cloud_datum.json) and run the graph. Three millimeter
scans share a rotated fixture pose. The inspection uses an 80 × 80 mm
footprint with a 4 × 4 grid, a nominal height of 2 mm, and a ±1 mm height
band. The coverage limit is 97%. The peak-to-valley limit is 3 mm.

| Scan | Mean height | Datum-parallel peak-to-valley | Coverage | Expected state |
| --- | ---: | ---: | ---: | --- |
| Nominal | 2 mm | Near zero | 100% | Pass |
| Raised | 6 mm | Near zero | 100% | Fail height |
| Warped | 2 mm | 6 mm | 93.75% | Fail height, coverage and spread |

The graph contains two intentional failures. The headless runner exits
with status 1 and a report can record the separate outcomes.

**Deviation** is the signed distance from nominal in meters, with NaN
where coverage is missing. Each supported cell shows the sample with the
largest absolute deviation. **Coverage mask** shows the cells meeting
the configured minimum count. **Defect mask** marks a cell if any observed
height is outside the band, including cells with too few points for a
deviation map value. **Defects** retains the original point rows and their
attributes. All sampled points inside the footprint contribute to the
statistics and tolerance check.

For a fixture pose, local XY gives the footprint axes and +Z gives positive
height. For a fitted plane, the supplied normal gives positive height.
Map columns increase in U and rows decrease in V. Datum-parallel
peak-to-valley is the observed maximum height minus the minimum height.

The scans and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
