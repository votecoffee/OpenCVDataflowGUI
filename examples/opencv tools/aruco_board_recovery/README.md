# Recover a partially visible ArUco board

Open the [example project](aruco_board_recovery.json) and run once. The
graph compares direct marker decoding with geometry-guided recovery on the
same calibrated image. Open the result images to see decoded markers in
green, the recovered marker in amber and the ignored foreign marker in red.

The six-ID board has four clean markers and one damaged data module on
marker 7. A foreign marker numbered 42 covers the position of marker 12.
Both branches decode IDs 3, 18, 1 and 22 and ignore 42. Recovery adds ID 7
and leaves ID 12 missing. For the five-marker fit, the reprojection RMS is
0.240 px and the translation differs from synthetic truth by 0.262 mm.
See [the reference geometry](assets/truth.json).

For a real board, connect the target record and a calibrated pinhole camera
that matches the capture. Choose the seed and total marker counts, pixel
limits and recovery correction rate for the required pose quality. Output
matches follow the board's ID order and carry canonical pixel and physical
corners. A recovered match also gives its original rejected-candidate
index, starting at zero. Duplicate board IDs or a fit outside the configured
limits withhold the pose. A mask or region narrows the eligible marker centers.

OpenCV 5.0 requires the recovery bit distance to be below the whole-bit
limit obtained from the dictionary budget and the selected correction rate.
A rate of zero yields no recovered candidates.

The capture and ground truth are original synthetic fixtures by
OpenCVDataflowGUI contributors under the project's MIT license. Run
`build_example.py --output` pointing to an empty directory to regenerate
the example. [OpenCV ArUco detector documentation](https://docs.opencv.org/5.0/main_modules/classcv_1_1aruco_1_1ArucoDetector.html)
