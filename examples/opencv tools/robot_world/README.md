# Robot-world and hand–eye calibration

Open either saved project and run it once. Both examples work offline with the
included synthetic ChArUco views and robot poses.

| Project | Arrangement | Mount output | Station output |
| --- | --- | --- | --- |
| [Fixed camera](fixed_camera.json) | The camera watches a target attached to the gripper. | Target to gripper | Camera to base |
| [Robot-mounted camera](robot_mounted_camera.json) | The camera rides on the gripper and watches a stationary target. | Camera to gripper | Target to base |

Select the calibration block to inspect the two rigid transforms and the report.
The block fits both transforms from 12 views and checks closure on four separate
views. A failed fit or held-out RMS limit leaves both transform outputs empty and
returns the report with failure reasons.

| Held-out check | Fixed camera | Robot-mounted camera |
| --- | ---: | ---: |
| Translation closure RMS | 0.095 mm | 1.274 mm |
| Rotation closure RMS | 0.085° | 0.100° |
| Worst per-view image reprojection RMS | 0.123 px | 0.462 px |

The saved projects gate translation at 1.5 mm and rotation at 0.4°. A separate
regression check redetects ChArUco corners in the four included PNGs and projects
the board through each fitted solution. The two projects use the same target
images with robot poses generated for the respective mounting arrangements.
These tolerances and results apply to the synthetic fixture.

For recorded data, connect paired lists or ticketed groups of gripper-to-base
and target-to-camera poses. Supply three to 64 training pairs and optionally
one to 64 separate held-out pairs. Use consistent millimeters or meters and
matching frame names. The example loads the same inputs from project-relative
JSON files. The solver uses Park relative-motion initialization and SciPy
least-squares refinement of both transforms.

The original board renders, poses and [generator](build_example.py) are by the
OpenCVDataflowGUI contributors under the repository [MIT license](../../../LICENSE).
For the transform conventions and build details, see the
[3D guide](../../../docs/vision-3d.md#robot-world-calibration).
