# SimplePathWidth · Fixture pose replay

Open the [project](project.json) with SimplePathWidth 0.1.0.dev3 or later.
Run once, then inspect **Result / overlay**. The control-point transform
supplies the pose used to rotate the weld 8° and move it down 35 px.
SimplePathWidth uses that transform to place the trained path and its
cross-scans on the moved image.

Disconnect the **pose** wire and run again to compare the fixed path.
The demonstration recipe reports pass with the supplied pose and fail
with the fixed path. Both runs measure 31 stations. Mean width error
against the supplied mask falls from 11.3 px to 4.3 px with the pose.
The configured spacing is 8.01 px to retain 31 stations with the small
rounding error of the fitted transform.
For an actual installation, locate stable fixture or part features outside
the inspected bead so an off-path bead remains measurable as an offset.

The saved recipe sets **Edge continuity** to 2. It uses nearby stations to
select edges within the two search bands. Open **Train path width…** to
review the reference and change the setting.

The moved image was made from `20231113_19h46m30s_B13_3.png` by converting
it to grayscale, then applying the rotation and translation with bilinear
interpolation and replicated borders. The training image in `weld.spw`
is a grayscale copy of `20230930_21h03m42s_B11_1.png`. Both source images
are from Hyun Soo Kim (2024), *Weld bead images and masks*, Figshare v1,
file `46550428`. [Dataset](https://doi.org/10.6084/m9.figshare.25904014.v1) ·
[MIT license notice](LICENSE-data.txt)
