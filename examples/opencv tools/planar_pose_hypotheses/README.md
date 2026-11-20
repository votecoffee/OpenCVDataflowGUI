# Planar pose hypotheses

Open [planar_pose_hypotheses.json](planar_pose_hypotheses.json) and run once.
Both branches detect ChArUco corners in rendered images of the same target
and pass them to **Planar pose hypotheses** using the same camera model.
Select a pose block to view the two plane normals and their reprojection scores.

| View | Best RMS | Alternate RMS | Gap | Single pose |
| --- | ---: | ---: | ---: | --- |
| Near-frontal | 0.126 px | 0.220 px | 0.093 px | Withheld at the 0.20 px ambiguity threshold |
| Oblique | 0.082 px | 4.899 px | 4.817 px | Emitted |

The near-frontal alternatives differ by 3.88° despite their close reprojection
scores. The oblique selection differs from the rendering pose by 0.068 mm at
the board origin and 0.027°. These tolerances and results describe the included
synthetic captures. The [generator](build_example.py) records both reference
poses. The images and project are original fixtures by the OpenCVDataflowGUI
contributors under the repository [MIT license](../../../LICENSE).

For recorded data, provide matched coplanar object coordinates and distorted
image coordinates from a calibrated pinhole camera. IPPE accepts at least
four non-collinear pairs. IPPE square uses exactly four object corners on a
square centered at the origin with Z=0. Enter the corners in top-left,
top-right, bottom-right, bottom-left order with object +Y pointing up.
[OpenCV PnP methods](https://docs.opencv.org/doc/doxygen/html/d5/d1f/calib3d_solvePnP.html)

**Hypotheses** retains the distinct positive-depth candidates that meet the
reprojection limit, ranked by pixel RMS. **Ambiguous** flags a close score
gap. The activation passes when it found plausible hypotheses. A path that
requires a single transform should consume **Pose**, which emits only when
one candidate meets the limits or the leading candidate clears the gap.
**Report** includes the RMS and rejection status for each native candidate.
