# Image feature evaluation

**Evaluate image features and matches** (`inspect.evaluate_features`) measures
center-based detector repeatability and descriptor-match precision/recall against
a caller-supplied query-to-train homography. It reuses feature sets from the
existing ORB, BRISK, AKAZE, SIFT and KAZE blocks and can evaluate existing matcher
outputs. Find it under **OpenCV → Features Framework (features) → Feature evaluation**.

The installed OpenCV 5 Python wheel does not expose `evaluateFeatureDetector` or
`computeRecallPrecisionCurve`. This block provides an explicitly defined
center-reprojection benchmark using native projection/matching and a validated
one-to-one assignment. OpenCV's legacy ellipse-overlap benchmark is a separate
metric; its native numerical results should not be compared as identical scores.

Try the [eight examples](<../examples/opencv tools/feature_evaluation/README.md>).
Their rendering transforms are declared independently of feature extraction and
matching. No plugin, model download, optional package or camera is needed.

## Inputs and reference geometry

Connect **Query image**, **Train image**, and their corresponding **Query/Train**
feature outputs. Descriptor families/settings must agree. Native binary families
use Hamming (ORB WTA_K 3/4 uses Hamming2); SIFT, KAZE and float AKAZE use L2.
Native feature blocks retain all of their own detector controls.

**Transform** overrides Reference transform, otherwise the configured 3×3 matrix
is used. It must map query-image pixels to train-image pixels. Optional declared
source/target image sizes are checked. Matrices are normalized, finite,
invertible, limited to condition number 1e12, and must have no projective horizon
crossing either image. The tool never fits a transform. A held-out rendering
matrix, independent calibration or independently measured registration is suitable
truth. Fitting the evaluated matches themselves measures agreement with that fit
and can bias the result. Independence is recorded as caller supplied/unverified.
One planar homography does not model arbitrary nonplanar scene motion.

Feature records must match image dimensions, center bounds and descriptor layout.
Known feature/image frames and acquisition tickets must agree within each side;
query and train may have different frames/tickets. Their contexts remain in
Metrics. Matching metadata cannot prove physical exposure synchronization or that
a supplied transform is ground truth. Images/points from different pyramid grids
need explicit upstream conversion/projection before evaluation.

Query/Train mask and region inputs intersect separately on their own image grids.
They select keypoint centers with the common nearest-pixel mask convention.
Border margin excludes centers near image boundaries. A query center is eligible
when it and its forward projection are in both selected images. Train eligibility
uses the inverse projection. This defines common observable support. Full keypoint
patches, scale, orientation and support-ellipse overlap are not evaluated.

## Metrics and thresholds

1. A geometric edge links eligible centers when forward projection error is
   at most Reprojection tolerance in **target-image pixels** (default 3).
2. Maximum-cardinality one-to-one pairing of these edges gives the correspondence
   count. An augmenting-path algorithm avoids the undercount possible with a
   nearest-first greedy assignment.
3. Repeatability is correspondence count divided by the smaller eligible feature
   count. Feature rows remain distinct, including repeated scale/orientation
   detections; the count is not a physical part count.
4. At each inclusive ranking threshold, accepted candidate rows are scored against
   those geometric edges. True positives are a maximum one-to-one subset of
   correct accepted pairs. Precision is true positives / accepted rows; incorrect
   rows and redundant duplicates consume precision without adding true positives.
   Recall is true positives / geometric correspondence count. Alternative valid
   assignments are accepted, rather than requiring one arbitrary canonical pair.

With no Matches input, native brute-force matching searches the eligible feature
pool and produces one nearest candidate per eligible query. Choose descriptor
**distance** or nearest/second-nearest **ratio** ranking. Missing second neighbors
or zero second distance produce explicitly unranked rows; no arbitrary ratio is
invented. Lower ranks are better; distances are family-specific, not probabilities.

The optional **Matches** input accepts the indexed paired record from existing
matchers, including radius matching. Its distances always become ranks. Image
sizes, row indices and paired coordinates are checked against the feature sets.
Rows outside common support are excluded and counted explicitly. Sparse input
candidates limit achievable recall; they are not presumed exhaustive.

The precision/recall table groups equal ranks together. Each reported point is
exact at its threshold. Maximum curve points samples thresholds when necessary,
always retaining the operating threshold and highest observed eligible rank.
It does not report an average-precision integral from the sampled curve.
The diagnostic plot displays those sampled precision/recall pairs.

Minimum correspondences, repeatability, precision and recall control inspection
judgment. Zero minimum precision/recall disables that particular criterion.
No common support or no geometric correspondence always fails. Undefined values
are explicit (`Metrics.valid`, null table values, and no scalar output event),
not NaN or fabricated perfect scores. A valid benchmark with disabled match-quality
criteria can pass while reporting zero accepted matches. Configure minima for the
inspection decision you intend. Block score is repeatability, not model confidence.

## Outputs, graphics and limits

| Output | Meaning |
| --- | --- |
| Metrics | Counts, eligibility, validity, ranking, operating confusion counts, contexts, reference matrix, limits/criteria reasons and curve-sampling flag. |
| Repeatability / Precision / Recall | Optional scalars; undefined metrics emit no event. |
| Correspondences | One maximum geometric assignment with original feature indices and target-pixel errors. |
| Matches | Diagnostic rows: original indices, distance/rank, reprojection error, eligibility, geometric validity, operating acceptance and unique true-positive flag. |
| Curve | Threshold, accepted/true/false-positive/false-negative counts and precision/recall. |
| Canvas / Overlay | Optional side-by-side diagnostics with their own grid. Metrics supplies each source-to-canvas map. |

Green lines show unique correct accepted pairs; red lines show other accepted
rows. Cyan marks show query reprojections on the train side. Drawable geometry is
anchored to the comparison canvas, not either original camera grid. The precision/
recall plot is a separate diagnostic raster. Global/per-block graphics-off skips
both images, drawings and plot; numerical results remain available. Source images
and descriptor tables are not modified. Preview dimensions and Maximum graphic
matches bound display work independently of evaluation counts.

Default limits are 2,000 features per side, four million full descriptor pairs,
100,000 geometric edges, 10,000 candidate rows, 64 curve thresholds and 8,388,608
pixels per image. Parameter limits also bound diagnostic dimensions. Oversized or
invalid inputs produce an error with no current outputs. Caps do not guarantee
native allocation sizes, process memory or execution time; common runtime/editor
copies remain outside these working-set limits. See all
[controls](block-parameters.md#evaluate-image-features-and-matches-inspectevaluate_features).

The common executor owns timing, logging, tickets, exception containment and
downstream failure handling. Native detector/matcher errors cannot escape this
boundary; a later valid invocation can recover. Independent instances support
parallel branches and isolated previews. JSON/XML retain configuration, including
the reference matrix and criteria, without a trained or accumulating model.

Native references: [Features Framework](https://docs.opencv.org/5.0/main_modules/features.html)
and [legacy evaluation definitions](https://github.com/opencv/opencv/blob/5.0.0/modules/features/src/evaluation.cpp).
