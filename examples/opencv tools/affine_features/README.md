# Affine-invariant image features

Open `project.json` and run once. No plugin, model download or optional dependency
is needed. The two original synthetic images represent a textured inspection
plate and its known X-compressed, sheared view.

The graph extracts affine SIFT features in both images, uses the existing SIFT
matcher, estimates a homography, and displays paired points. The review image
shows green inliers and red outliers against the fitted homography. Each feature
block also draws its source-image support ellipses and orientation directions.
Its `image` output retains the original pixels, encoding and depth.

The saved configuration samples 18 views per image. On the pinned OpenCV 5 build,
it produces 472 descriptor correspondences, with 97.9% within 3 pixels of the
independent transform in `assets/truth.json`. Median truth error is 0.71 pixels;
the fitted transform's maximum error at four interior test points is 0.56 pixels.
There are 236 distinct 5-pixel source cells among the accurate correspondences.
Repeated affine views can describe the same physical feature more than once;
these counts are not counts of parts. Numerical details can vary across builds.

From the repository root:

```powershell
opencv-dataflow-run "examples/opencv tools/affine_features/project.json"
```

To regenerate the original fixtures and graph:

```powershell
python "examples/opencv tools/affine_features/generate.py"
```

## Controls and coordinates

**Affine-invariant image features** offers SIFT and ORB backends. Their descriptors
connect to the ordinary SIFT or ORB matchers, respectively. All backend controls
are persisted; controls prefixed with the inactive backend's name are unused.
SIFT byte descriptors still use L1/L2 distance. ORB WTA_K 3/4 uses Hamming2 in
the existing ORB matcher.

Generated views use `tilt_step ** index`, from the minimum to maximum tilt
index inclusive. Index zero samples the original image once. Other levels use
roll spacing `rotate_step_base / tilt` below 180 degrees. Custom mode instead
accepts explicit `[tilt_factor, roll_degrees]` pairs. Limits are 64 views, tilt
factors up to eight, 4 MP input images and 64 MP summed rotated-canvas work.
Requests for more than 64,000 detector features across all views are rejected.
The final feature limit keeps strongest selected responses and matching descriptor
rows together. These are workload guards, not a guaranteed native memory ceiling.

Masks and circle/rectangle/polygon regions are intersected in the source frame.
Only selected source centers survive. Source-frame and ticket/pyramid payloads
remain on the output wires. Coordinates stay in the selected input image's
pixels; a pyramid level is not silently converted to full-resolution pixels.

OpenCV maps keypoint centers back from each virtual view but retains view-space
size and angle. This block stores those originals as `view_size`, `view_angle`
and `view_index`. Its ordinary `size` is the source-image equal-area diameter;
`angle` is the mapped source-image orientation, or -1 if unknown. The exact
source support ellipse and orientation radius are stored alongside them.
Octave indices remain local to the virtual view. See the
[OpenCV 5 AffineFeature API](https://docs.opencv.org/5.0/main_modules/classcv_1_1AffineFeature.html)
and [native implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/features/src/affine_feature.cpp).

Graphics disabled skips ellipse drawing and the optional preview; feature
extraction and original-image passthrough remain available. Affine sampling is
more expensive than a single detector call. This synthetic repeatability test
does not qualify real perspective, occlusion or illumination changes.

## Attribution

The generator, images and project are original repository examples, distributed
under the repository's MIT license. No customer images or third-party image
assets are included.
