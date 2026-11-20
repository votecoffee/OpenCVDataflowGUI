# Marked and oversized checkerboards

Open `marked_checkerboard.json` and run the graph. The first branch detects
an exact 17 × 13 inner-corner grid. The next two detect different parts of
the same oversized target, including a rotated perspective view. Compare
the signed `corner_ids` and the origin and axis overlay in Debug. Identical
IDs refer to identical physical positions across the views.

The last branch deliberately crops out the origin markers. It reports a
failed detection and emits no board coordinates, so the headless example
exits with an intentional failure.

The pattern has three dots on adjacent cells. A black dot marks the white
origin cell, with white dots on the black cells to its right and above
in the canonical front view. Board `[0,0]` is the upper-left **intersection**
of the black-dot cell in that view. OpenCV evaluates cell metadata in its
own detector order. The block preserves the raw metadata and converts the
detected intersections to a consistent physical order. Set **Spacing** to
the measured corner pitch and use the same target ID for every capture.
The image and object points can feed **3D object pose (PnP)** with the
matching camera calibration. Use front-face captures with normal image
handedness. The marker geometry alone cannot identify a reflected source.
[OpenCV pattern guide](https://github.com/opencv/opencv/blob/5.x/doc/tutorials/calib3d/camera_calibration_pattern/camera_calibration_pattern.markdown)

The images are original synthetic test fixtures by OpenCVDataflowGUI
contributors under the repository’s MIT license. The generator can recreate
the project and images with `build_example.py --output <empty directory>`.
