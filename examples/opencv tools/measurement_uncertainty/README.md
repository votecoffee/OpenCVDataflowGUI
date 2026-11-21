# Calibrated measurement uncertainty

Open [measurement_uncertainty.json](measurement_uncertainty.json) and run
once. The graph projects a known segment from a tilted board into a distorted
pinhole image. **Measure planar distance uncertainty** recovers the two
endpoints on the board and estimates a distance of **144.838 mm** with
standard uncertainty **0.290 mm** and a central 95% interval of
**144.269 to 145.406 mm**. Debug mode draws the measurement on the original
synthetic image.

## Supply your own uncertainty model

1. Connect the pinhole camera and the rigid pose from the local measurement
   plane (Z = 0) to the camera. Supply two endpoints in the camera's
   distorted pixel grid, and optionally the corresponding image.
2. Set **Pixel stddev px** for independent pixel coordinates, or provide a
   named covariance matrix in **Input covariance** or through the covariance
   connector. For example, this matrix represents 0.5 px standard deviation
   at each endpoint and a correlation coefficient of 0.8 between their X
   coordinates.

   ```json
   {"terms": ["u0", "u1"], "matrix": [[0.25, 0.20], [0.20, 0.25]]}
   ```

   Named pixel terms override the independent pixel setting for those
   coordinates. Omitted model terms are held fixed. Use `u0,v0,u1,v1` in
   pixels, `rx,ry,rz` as additive components of OpenCV's Rodrigues rotation
   vector in radians, `tx,ty,tz` in camera length units, and `fx,fy,cx,cy`
   in pixels. Distortion terms follow the supplied OpenCV lens model
   (`k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tau_x,tau_y`). The tilt terms
   use radians. Each covariance entry uses the product of its two term units.
3. Inspect the interval and the endpoint covariance in `[x0,y0,x1,y1]`
   order. Set **Maximum expanded uncertainty** to flag intervals with an
   excessive half-width. Geometry at the plane horizon, invalid
   intersections and separations unresolved at the declared uncertainty
   withhold the measurement outputs.

The interval uses a local Gaussian approximation and the OpenCV projection
Jacobian. Both endpoints share the pose and lens terms. Common-mode errors
can cancel in the distance. The model assumes small perturbations, the
specified plane, and any explicitly declared covariance. Fit RMS alone
does not specify covariance or a measurement tolerance.

## Verification

- In 24,000 seeded synthetic perturbations, an independent Brown projection
  and ray-intersection model gave 0.290 mm standard deviation and **95.1%**
  empirical coverage for the configured 95% interval.
- The recorded check fits intrinsics on six OpenCV sample photographs and
  evaluates five separate photographs. Each pose fit excludes the two
  corners used for measurement. All five intervals included the nominal
  eight-square spacing. Observed standard deviation was 0.0147 square
  pitches, compared with 0.0101 predicted by the pixel-only assumption.
  The camera and plane uncertainties were held fixed for this comparison.
  The square pitch is a nominal unit, with no independently measured
  dimensional reference. [OpenCV 5.0.0 sample images](https://github.com/opencv/opencv/tree/5.0.0/samples/data)

Run the recorded check against a local copy of the pinned sample directory.
The check verifies image digests and uses the supplied files.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/measurement_uncertainty/check_recorded.py" <opencv-5.0.0-samples-data>
```

The supplied synthetic fixture is original and uses the repository's MIT
license. Rebuild it in an empty directory.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/measurement_uncertainty/build_example.py" --output <scratch-directory>
```
