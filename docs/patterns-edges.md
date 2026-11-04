# Pattern training and oriented edges

## Example

Open **File > Load trained patterns and edges example** and press F5, or open
`examples/trained_patterns.json`. Synthetic reference images train two patterns,
which are located in a larger scene. A rectangular region isolates a vertical bar.
Expected results are:

| Tool | Result |
| --- | --- |
| Geometric matcher | One match centered near `(125.5, 127.5)`, rotated +10 degrees |
| ORB texture matcher | One match centered near `(425.5, 215.5)` |
| Oriented edges | Two strong vertical edges, each over 250 pixels long |

The project has no camera, external asset or output writer. Verify repeated runs:

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/trained_patterns.json --repeat 3
```

## Train once and reuse

Add **Inspection > Pattern matching > Match shape pattern** or **Match texture
pattern**. Connect an image and run once to capture it. An untrained matcher reports
an error but retains its input for training. Select the matcher, configure detector
controls, and click **Train from input** in the inspector. Run again to search.
The model is saved inside the project, including nested ToolBlocks, and copied into
background execution. Training does not happen automatically on subsequent runs.

Training uses the captured image and optional `training_mask` / `training_region`.
If those are absent, the ordinary `mask` / `region` ports select the training area.
Masks and regions combine by intersection. Regions support rectangles, circles and
polygons. The selection's bounding box defines the model-local image dimensions;
only selected features are retained. An empty selection or insufficient features
rejects training and preserves the previous model. First run upstream region/mask
blocks to capture their values, then train. Training from the inspector is a
synchronous operation; restrict large inputs to a useful training area.

During search, `mask` / `region` restrict candidate anchor centers. Texture matching
also restricts detected target features. Shape matching may use neighboring features
outside that selection; it does not require the entire projected pattern to be
inside it. Training-only ports do not restrict runtime search.

**Load trained model**, **Save trained model** and **Clear trained model** manage
standalone JSON models. Export requires a new filename. Imports check model kind,
dimensions, feature data and content checksum before replacing the current model.
This checksum detects accidental edits; it is not an authenticity signature.

For graph-driven training, use **Inspection > Pattern training > Train shape
pattern** or **Train texture pattern** with a reference image and optional mask/
region. Connect its `model` output to any compatible matcher. These trainer blocks
retrain on every execution. A connected model takes precedence over an embedded
model and disables inspector model editing. Encapsulate the trainer and matcher
into a ToolBlock to reuse the complete pattern.

## Geometric matching

`pattern.match_shape` trains from edges, corners, Hough lines, Hough circles, or their
combination. Detector thresholds, kernel/aperture sizes, corner/Harris controls,
line voting/gaps/length and circle radius/threshold controls are exposed. The model
retains sampled feature points and its extraction settings. Changing training
controls requires retraining; the saved model's settings are used for search.

Search evaluates explicit rotation and scale samples. Set minimum, maximum and
step for each. The defaults search translation at the trained scale and angle.
Positive rotation follows OpenCV's counterclockwise convention. Chamfer distance
measures trained feature distance to the nearest detected target feature:
`score = exp(-mean_distance / distance_tolerance)`. Coverage is the fraction within
that pixel tolerance. Minimum score and coverage must both pass. `max_matches` and
`max_overlap` control multiple-instance output and bounding-box IoU suppression.

This is sampled geometric localization, with no learned classifier or unrestricted
deformation search. Extra scene edges are not penalized directly; repetitive or
dense texture can create ambiguity. Restrict the search region and tune thresholds
on representative images. Models are bounded to 4096 pixels per axis; search is
limited to 128 rotation/scale combinations and 100 million image-pixel variants.

## Texture matching

`pattern.match_texture` trains ORB or SIFT keypoints/descriptors. ORB exposes feature
count, pyramid scale/levels, edge/patch sizes, WTA_K, score mode and FAST threshold.
SIFT exposes count, octave layers, contrast/edge thresholds, sigma and precise
upscaling. Algorithm and enum options use dropdowns.

Runtime matching uses the trained detector settings, the appropriate Hamming/
Hamming2 or L2 distance, a nearest-neighbor ratio, optional mutual consistency and
maximum descriptor distance. Similarity, affine or homography pose estimation
supports RANSAC/LMEDS, reprojection threshold, iterations and confidence; affine
models also expose refinement iterations. Quality gates include correspondence/
inlier counts, inlier ratio, feature coverage, RMS reprojection error and scale.
The score is inliers divided by accepted correspondences. Model coverage is
inliers divided by trained feature count, not physical area coverage.

One best planar pose is returned per matcher. This does not yet find every repeated
texture instance or handle nonrigid/3D pose. Reflected, nonconvex, degenerate or
out-of-range poses are rejected. A valid image with no accepted pose is a normal
inspection failure with empty matches, zero count and zero score. Missing/invalid
models or controls are execution errors and clear stale results.

## Results and coordinates

Both matchers return `matches`, `count`, `score`, `transform`, center `points` and
an `overlay` image, plus an offset `pose`. Each match includes center, corners, rotation, scale, score and
method-specific quality measurements. `transform` is the best match's 3-by-3
`source_to_target` matrix; no match yields `{}`. Image inputs are uint8 grayscale
or supported uint8 color converted to grayscale.

The source frame is model-local, with origin at the top-left of the training
selection's bounding box. The model stores `training_origin` separately. Pose
outputs include source and target coordinate payloads; **Transform points** can map
model-local points into the actual search-image frame. Pyramid-level image
selection works through the existing input-level controls, and target payloads
retain level/root provenance. Numeric coordinates refer to the selected image
level; use coordinate mapping when comparing to full-resolution results.

`pose_offset_x`, `pose_offset_y` and `pose_offset_angle` define a new inspection
frame relative to the trained rectangle's center. Offsets are local model pixels
and follow the fitted rotation/scale/projective mapping. Original match centers,
corners and `transform` remain unchanged. Every match record includes its offset
`pose`; the pose port returns the best one, or `{}` on no match. See
[pose-relative inspection](poses.md) for feature-based frames and moving regions.

## Oriented edge finding

**Inspection > Edges > Find oriented edges** (`inspect.oriented_edges`) uses
Gaussian smoothing and normalized Sobel gradients on uint8 grayscale/color inputs.
Choose vertical, horizontal or angled tangent direction. Tangent angles are in
image coordinates: 0 horizontal, 90 vertical, positive towards increasing y.
`angle_tolerance` gates both local gradients and the fitted line direction.

Polarity is rising or falling along the normal `(tangent - 90) mod 180`: rightward
for vertical edges and downward for horizontal edges. Either polarity reports
the two sides of a bright stripe separately. Controls include:

| Control | Meaning |
| --- | --- |
| Response threshold | Minimum per-pixel projected gradient, in gray intensity/pixel |
| Strength bounds/statistic | Mean, median or peak projected gradient over each edge band |
| Thickness bounds | Width of the thresholded gradient band normal to its fitted line, in pixels |
| Minimum length | Total fitted span, including joined gaps |
| Minimum continuous length | Longest original connected run; joining gaps never increases it |
| Maximum gap/normal distance | Optional joining of separated, approximately collinear runs |
| Minimum coverage | Union of original run spans divided by total span |
| Intensity bounds | Mean original grayscale intensity at measured edge pixels, 0..255 |
| Contrast bounds/offset | Absolute mean difference sampled on opposite sides beyond half band width plus offset |

Zero maximum strength, thickness or contrast means no upper bound; zero maximum gap
disables joining. Thickness measures the edge's transition band, **not the width of
a bright bar**. Blur and response threshold affect this width. Contrast samples
need both sides inside the image; unavailable contrast fails an enabled contrast
gate. Sampling uses the original grayscale image, including neighboring context
outside a selection. Masks/regions clip measured pixels, not the gradient context.

Outputs include line `collection`, endpoint `segments`, count, accepted-pixel mask,
float32 response image and overlay. A valid empty result has count zero and remains
a successful measurement. The line collection carries image coordinate provenance
and supports existing collection filtering/sorting by thickness, continuous length,
strength, intensity, contrast, coverage and ordinary line geometry. Gap pixels are
not filled in the accepted mask. Curved edges may split into several detections;
this tool measures locally straight gradient bands, not complete object boundaries.

Native references: [OpenCV feature detection](https://docs.opencv.org/4.x/d5/d51/group__features2d__main.html),
[calibration and robust transforms](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html),
[image gradients](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html).
