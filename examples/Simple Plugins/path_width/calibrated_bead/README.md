# SimplePathWidth · Calibrated bead

The [project](project.json) uses a generated view of a 6 mm wide bead.
The bead center is 1.3 mm from the taught path on its positive side.
A projective transform tilts and skews the view. The saved planar
calibration contains the known inverse transform.

Use SimplePathWidth 0.1.0.dev5 or later. See the [package setup](../README.md#setup).
Run the project and select **Measure bead**.

| Output | Expected result |
| --- | --- |
| **overlay** | The image-space path and paired edge calipers |
| **trend** | Pixel widths vary from about 30.5 to 37.8 px |
| **physical_trend** | Plane widths stay near 6 mm and signed offset near +1.3 mm |
| **physical_measurements** | Units, calibration fit RMS, path length, and per-station plane coordinates, widths, and offsets |
| **mean_width_physical** | About 6.03 mm |

**Load calibration profile** supplies the mapping in the inspection image's
pixel coordinates. For a real camera, create a planar profile with a
calibration target and use the same camera and preprocessing coordinates.
The width calculation uses the plane normal to the path. The edge search,
gap length, and pass/fail thresholds use pixels. This example uses exact
synthetic geometry to demonstrate conversion with a supplied mapping.
