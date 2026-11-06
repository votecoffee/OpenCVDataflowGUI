# ChArUco board

Open [charuco_board.json](charuco_board.json) with **File → Open…**
and click **Run once**. The target has 7 by 5 squares and is tilted
and cropped. **Detect ChArUco board** identifies 17 of its 24 corners.
The PnP block recovers the pose from the visible points with about
0.16 px reprojection RMS. The overlay labels the ArUco markers and
the identified ChArUco corners.

Find **Detect ChArUco board** under **Calibration → Grid and coordinates**.
Set rows and columns to the number of squares. Enter square and marker
side lengths in the same units. Set the dictionary, legacy layout,
and custom IDs as needed to match the board. The example uses 30 mm
squares and 22 mm markers, with lengths configured in meters.

**Observation** records the board definition, visible marker IDs, and
matched points. **Image points** and **Object points** are ordered by
the same ascending **Corner IDs**. They emit when enough noncollinear
corners are visible. The optional camera input lets OpenCV use pinhole
intrinsics to interpolate corners. Use the same length units for the
board and camera pose.

```powershell
opencv-dataflow-run examples/charuco_board/charuco_board.json
```

[OpenCV ChArUco detection](https://docs.opencv.org/5.0/tutorials/objdetect/charuco_detection/charuco_detection.html) · [All examples](../README.md)
