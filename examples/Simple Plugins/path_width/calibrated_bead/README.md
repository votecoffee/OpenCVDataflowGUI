# SimplePathWidth · Calibrated bead

The [project](project.json) replays three generated beads through the same
tilted view. The saved planar calibration contains the known inverse
transform. The recipe accepts widths of 5.8–6.2 mm and absolute center
offsets up to 1.5 mm.

Use SimplePathWidth 0.1.0.dev6 or later. See the [package setup](../README.md#setup).
Run once per frame and select **Measure bead**.

| Frame | Mean width | Approximate center offset | Verdict |
| --- | ---: | ---: | --- |
| Nominal bead | 6.03 mm | +1.3 mm | Pass |
| Narrow bead | 5.22 mm | +1.3 mm | Fail, below minimum width |
| Off-path bead | 6.01 mm | +2.1 mm | Fail, offset exceeds the limit |

| Output | Expected result |
| --- | --- |
| **overlay** | The image-space path and paired edge calipers, red where a station fails |
| **trend** | Nominal bead widths vary from about 30.5 to 37.8 px |
| **physical_trend** | Plane widths and signed offsets with physical tolerance lines |
| **physical_measurements** | Units, path and gap lengths, per-station dimensions and failure reasons |
| **mean_width_physical**, **longest_gap_physical** | Scalar results in the profile units |

**Load calibration profile** supplies the mapping in the inspection image's
pixel coordinates. For a real camera, create a planar profile with a
calibration target and use the same camera and preprocessing coordinates.
To edit the limits, open **Train path width…** and scroll to **Physical
units** and the physical tolerance fields. Reference preview uses pixels.
At runtime the physical limits require a profile with matching units.
Any configured pixel limits still apply. The example uses exact synthetic
geometry to demonstrate measurement and judgment with a supplied mapping.
