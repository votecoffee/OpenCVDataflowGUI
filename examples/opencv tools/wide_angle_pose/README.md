# Wide-angle object pose

Open [fisheye_object_pose.json](fisheye_object_pose.json) or
[omnidir_object_pose.json](omnidir_object_pose.json) and run once. Both graphs
recover a known object-to-camera pose from noisy landmark coordinates in a
distorted image. Each retains all 54 valid pairs, rejects eight deliberately
mismatched pairs, and rejects one out-of-domain ray.

| Lens | Source RMS | Maximum error on five withheld landmarks |
| --- | ---: | ---: |
| Fisheye | 0.085 px | 0.041 px |
| Omnidirectional | 0.077 px | 0.028 px |

The measured XYZ and XY pairs are simulated inputs. The independent fixture
generator projects them with the fisheye or Mei camera equations.
[Ground truth](assets/truth.json) includes the transform, original row IDs
and withheld observations.

## Use the block

1. Wire a fisheye or omnidirectional **Camera intrinsic model**, paired
   object XYZ and raw image XY, and optionally the matching source image.
   Set the object frame and use the same length units for the object points
   and the camera. The output pose maps the object frame into the camera.
2. Choose the native solver and whether to use RANSAC. IPPE requires
   coplanar object points. The Iterative method accepts an optional
   initial object-to-camera pose. Fisheye uses OpenCV's native fisheye PnP.
   Omnidirectional uses standard PnP on rays from the Mei camera model.
3. The RANSAC proposal tolerance is in virtual pinhole pixels using the same
   intrinsics. Final inlier classification and optional refinement use the
   source lens and original distorted pixels. Inspect original row indices,
   reprojection errors, unsupported rays and the final RMS in the report.

This path supports forward-facing rays with a positive camera Z component.
The minimum ray cosine excludes the neighborhood of the horizon. The block
checks each inverse ray by projecting it back into the source lens. The
omnidirectional path requires nonnegative `xi` and a calibrated region with
a unique forward inverse. This baseline covers forward fields of view and
leaves rear-facing observations outside the supported domain.
[OpenCV fisheye solver implementation](https://github.com/opencv/opencv/pull/26669/files) ·
[OpenCV Mei projection reference](https://docs.opencv.org/5.0/extra_modules/ccalib.html)

The included fixtures are original and use the repository's MIT license.
Rebuild both projects and their assets in an empty directory.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/wide_angle_pose/build_example.py" --output <scratch-directory>
```
