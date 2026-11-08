# Registration, transforms and additional filters

These 12 blocks use eager OpenCV calls and the existing graph/result contracts.
Restart the app to see their toolbox entries. They do not change existing saved
graphs, layouts or parameters.

| Type / subtype | Block | Native operation |
| --- | --- | --- |
| Image processing / Geometry | Warp affine | `warpAffine`; JSON 2×3 input-to-output matrix or optional transform input |
| Image processing / Geometry | Warp perspective | `warpPerspective`; JSON 3×3 matrix or estimated image homography input |
| Image processing / Geometry | Rotate arbitrary angle | `getRotationMatrix2D` and `warpAffine`; angle, scale, center, post-rotation offsets and canvas mode |
| Image processing / Geometry | Add image border | `copyMakeBorder`; configurable widths and constant/replicate/reflect border |
| Image processing / Geometry | Gaussian pyramid up/down | Repeated `pyrDown` or `pyrUp`, 1–6 levels |
| Image processing / Filtering | Custom 2D filter | `filter2D`; finite JSON kernel, correlation/convolution, anchor, depth, delta and border mode |
| Image processing / Filtering | Unsharp mask | Gaussian blur plus weighted subtraction; kernel, sigma and amount |
| Inspection / Features | FAST corners | `FastFeatureDetector`; pattern, threshold and nonmax suppression |
| Inspection / Features | ORB keypoints and descriptors | `ORB.detectAndCompute`; keypoints, 32-byte descriptors, points and overlay |
| Inspection / Matching | Match ORB descriptors | Hamming `BFMatcher.knnMatch`, ratio test and maximum distance |
| Inspection / Registration | Estimate image homography | `findHomography` with RANSAC; paired points, inlier mask/ratio and pixel RMS |
| Image processing / Geometry | Transform image points | `perspectiveTransform`; optional inverse, matrix input and point list |

## Geometry contracts

Warp matrices describe **input/source pixels → output/target pixels**. The optional
transform port accepts a raw matrix or an estimated transform dictionary containing
`source_to_target`. Affine expects 2×3 and perspective expects 3×3. Matrices must
be finite and invertible. Output width or height of zero uses the source
dimension. The automatic rotation center is `((width-1)/2, (height-1)/2)`.
Positive angles follow OpenCV's rotation convention.

**Rotate arbitrary angle** applies the X and Y offsets after rotation in one warp.
**Same size** keeps the original canvas. **Fit** sizes and centers the full
transformed pixel footprint. Its `canvas_origin` reports the fitted grid's top-left
pixel center in the fixed-canvas coordinates, so changing the pivot or offsets
can change that origin while the fitted image stays centered.
Enable **Emit valid mask** to select pixels covered by the source using nearest
sampling. The mask excludes extrapolated border pixels. Other interpolation
modes can still blend across the source boundary.

Warp interpolation is nearest/linear/cubic. Borders are constant, replicated or
`BORDER_REFLECT_101`. Geometry, filters and pyramids preserve channel encoding
and source depth. Typical supported depths are uint8/uint16/float32/float64;
unsupported native combinations become block errors. Physical depth planes must
use dedicated depth tools. Per-channel filtering includes alpha if present.
Output allocations are limited to 64 million pixels.

Every geometry block returns an `output_to_input` matrix alongside its image.
Transform image points can consume that geometry directly to map output-image
measurements back to the source. It also accepts a `source_to_target` dictionary
or raw 3×3 matrix, and can invert the selected matrix. Empty point lists return
empty output; points mapping to infinity are rejected.

For example, wire Crop ROI's `geometry` to Transform image points' `transform`
and Connected components' `centroids` to its `points`. The resulting source-frame
points can then feed Pixel to physical using a calibration of that source frame.
Transforms are never silently chained into calibration metadata.
**Transform image points** waits for an emitted matrix when its transform port
is wired. The [subpixel crop example](<../examples/opencv tools/subpixel_crop/README.md>) shows
this with a fractional patch center.

Pyramids use OpenCV's Gaussian resampling, not a plain nearest resize. Their
coordinate transforms follow the level's sample origin with a factor of 2 per
down level, or 1/2 per up level. Odd-sized images use native pyramid rounding.

Custom filter output keeps the input depth, so unsigned data clips negative and
over-range filter results. Use the signed float gradient tools when you need raw
derivatives. Unsharp masking also follows native integer saturation. These filters
do not automatically correct gamma or change color spaces.

## Feature and registration contracts

FAST and ORB require uint8 input and interpret color encodings before grayscale
conversion. ORB outputs a JSON-compatible feature record with keypoint metadata,
descriptor rows, algorithm identifier and image size. Its binary descriptor uses
WTA_K=2 and 32 bytes. No features is a valid empty result.

Match ORB descriptors accepts two such feature records, applies the nearest/second
nearest distance ratio test and the maximum Hamming distance. There is no cross-check
or one-to-one assignment constraint. Fewer than two training descriptors produces
an empty match list. Malformed descriptor bytes or coordinates are rejected.
Matched `query_points` and `train_points` feed Estimate image homography.

The estimated matrix maps **query coordinates → train coordinates**. It requires
at least four paired, nondegenerate points and four RANSAC inliers. Its score is
the inlier fraction, not a probability of correctness; RMS uses the selected
inliers in target-image pixels. Minimum inlier count/ratio control inspection
pass/fail. A valid estimate below those acceptance thresholds still supplies a
matrix, following the existing inspection-failure contract. Check run status
before accepting a registration for measurement or downstream integration.

This is planar image registration, not camera intrinsics, physical calibration,
3D pose or robot extrinsics. Parallax, repeated texture and moving objects can
invalidate a fitted homography despite an apparently plausible image overlay.

## Lena registration example

Run `python scripts/prepare_lena.py` once, then open
`examples/opencv tools/lena_registration/lena_registration.json` in the editor, or run:

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/opencv tools/lena_registration/lena_registration.json" --repeat 3
```

The eight-block project creates an 8-degree rotated copy of Lena, extracts ORB
features from original and rotated images, matches descriptors, estimates the
original-to-rotated homography, warps the original, and shows an absolute-difference
image against the known rotation. The image input is project-relative and there
are no output writers. Try the rotation angle, ORB feature count, matcher ratio
and RANSAC tolerance. Counts vary with OpenCV versions and parameters.

Automated checks compare the recovered geometry to the known transform at several
image points; the workflow is a reproducible test, not a claim about registration
accuracy on unrelated camera scenes.

References: [OpenCV geometric transforms](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html),
[ORB](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html),
[calibration/homography](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
