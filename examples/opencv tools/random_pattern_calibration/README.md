# Random-pattern calibration target

Open [random_pattern_calibration.json](random_pattern_calibration.json) and
select **Run All**. The graph generates a 240 × 160 mm reference, reads two
images of that target, detects visible features and estimates its metric pose
with the known camera model. The detector's result image shows the fitted
boundary, accepted points and rejected matches.

The camera fields of view clip the target and an opaque rectangle covers part
of it in each image. The two observations share reference IDs. The supplied
[truth data](assets/truth.json) records the camera model and exact
target-to-camera poses, plus projections of 24 hidden target points. Those
points are inside the occluders and are evaluated independently of the fit.

## Use your own target

1. Set the physical width, height, units and raster width in **Generate
   random-pattern target**. The target record reports the raster size and
   effective DPI per axis.
2. Save the grayscale output losslessly and print at the configured width and
   height. Measure the textured rectangle edge to edge and use its physical
   dimensions. The sample print is [target.png](assets/target.png).
3. Wire the reference raster and target record to **Detect random-pattern
   target**. Pass the captured image to its image port. Use an optional mask or
   region to focus on the target. Supply the cropped or rectified image's
   camera model to a downstream pose solver, with matching length units.
4. Inspect the observation's inlier count, target hull coverage and pixel
   residuals. A failed view leaves the observation available and suppresses
   calibration point and homography connectors.

The detector uses OpenCV SIFT, reciprocal descriptor matching and a robust
homography. IDs are the row-major pixel indices of reference landmarks. XYZ
coordinates use the top-left outer edge as the origin, X right, Y down and
Z = 0. Record the lossless reference and physical dimensions when comparing
views. A single homography assumes a flat target and approximately pinhole
image geometry. Rectify substantial lens distortion before matching.

These original synthetic assets are contributed under the repository's MIT
license. The design follows the documented partial-view calibration workflow.
[OpenCV custom calibration pattern reference](https://docs.opencv.org/5.0/extra_modules/ccalib.html)

To regenerate the example, use the project interpreter and an empty output
directory.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/random_pattern_calibration/build_example.py" --output <scratch-directory>
```
