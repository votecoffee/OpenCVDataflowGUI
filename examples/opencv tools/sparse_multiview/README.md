# Sparse multi-view reconstruction

Open [sparse_multiview.json](sparse_multiview.json) and run once. The four
ChArUco camera views, their input poses, and the independent truth are included.
Views C and D have occluded corners and small position and orientation errors
in their supplied poses. Two reconstruction blocks see the same detections.
Select a reconstruction block to inspect the cloud and the per-landmark report.

| Result | Fixed poses | Joint refinement |
| --- | ---: | ---: |
| Landmarks recovered | 35 of 35 | 35 of 35 |
| Fit reprojection RMS | 0.775 px | 0.082 px |
| 3D landmark RMS against board geometry | 1.865 mm | 0.265 mm |
| Reprojection RMS on held-out view E | 0.662 px | 0.166 px |

Nineteen landmarks lack one of the four observations. The fifth image is
excluded from reconstruction and used by the regression check with its known
camera pose. Both metrics use the same reconstructed landmark IDs. These are
synthetic controls with known intrinsics and a calibrated baseline.

**Pack calibrated landmark view** accepts a camera model, a world-to-camera
pose, and matched ID and XY arrays. IDs must name the same landmarks in every
view. Combine the outputs with **Batch serial entries** as shown. The
reconstruction block also accepts a list on its connector or a JSON file
containing the same packed view records.

**Reconstruct sparse multi-view landmarks** accepts three to sixteen views.
It can triangulate without refinement, refine only the landmarks, or refine the
landmarks and camera poses jointly. In joint mode, set two trustworthy anchor
view IDs to fix the world frame and scale. Each adjusted view needs at least six
non-collinear landmarks seen from both anchors with sufficient parallax.
The report records observations, reprojection errors, and rejected landmarks.
[3D guide](../../../docs/vision-3d.md#sparse-multi-view-reconstruction)

The [fixture generator](build_example.py) creates the original rendered board
images, camera poses, and expected points. They are by the OpenCVDataflowGUI
contributors under the repository [MIT license](../../../LICENSE).
