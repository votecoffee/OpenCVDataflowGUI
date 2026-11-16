# Image feature evaluation

Open a project with **File → Open**, then **Run**. Select Evaluate image features
and matches to inspect Metrics, its side-by-side result and the precision/recall
plot. Change Match threshold and run the block using its last inputs to see how
the operating point changes. Source images and detector/matcher settings stay
available in their own nodes.

| Project | Expected outcome on the supported OpenCV 5 build |
| --- | --- |
| [Identity](identity.json) | Pass: repeatability, precision and recall are all 1 on the identical image. |
| [ORB rotation](orb_rotation.json) | Pass: compare detector repeatability and Hamming-distance matching across a known 8° rotation/translation. |
| [SIFT ratio](sift_ratio.json) | Pass: a known projective warp with nearest/second ratio ranking. |
| [Existing matcher](external_matches.json) | Pass at threshold 32: evaluate radius-matcher candidates. Increasing to 64 increases recall but lowers precision enough to fail its configured criterion. |
| [Selected support](selected_support.json) | Pass: target occlusion plus a left-side ROI; only centers observable in both selected images contribute. |
| [Wrong reference transform](wrong_transform.json) | Deliberate failure: the supplied transform has a 60-pixel error. The tool cannot establish that caller-supplied geometry is true. |
| [No features](empty_features.json) | Deliberate failure: no evaluable correspondences; undefined metric scalars emit no event. |
| [Pair budget](pair_budget.json) | Deliberate invalid-input error before native matching; no current outputs. |

The matrices in `truth.json` come from `build_example.py`'s image renderer,
independently of feature extraction or matching. They are never fitted from the
tested matches. Native feature counts can vary with OpenCV builds/settings.
Only the identity fixture promises exact 1.0 values; inspect the measured values
for the other known transformations and configured criteria.

This is a **center-reprojection** metric with maximum-cardinality one-to-one
pairing. The default tolerance is three target pixels. Full keypoint support
ellipse overlap, scale and orientation are separate metrics. Duplicated accepted
predictions reduce precision; valid alternative geometric assignments are allowed.
Both native legacy evaluation functions are absent from this Python wheel, so
this tool uses documented validated equivalents instead of claiming their exact
ellipse-overlap scores. See [definitions and limits](../../../docs/feature-evaluation.md).

Green/red lines distinguish unique true positives from other accepted candidates.
Cyan train-side marks show projected query positions. Canvas coordinates are
separate from either source image. Graphics-off/Lean skips diagnostic images
and the plot while preserving numerical results.

Run without the editor from the repository root:

```powershell
opencv-dataflow-api run "examples/opencv tools/feature_evaluation/orb_rotation.json" --summary
```

All images, projects and rendering matrices are original OpenCVDataflowGUI
contributor work under MIT, with attribution/comment blocks in every project.
No customer or third-party images are included. `truth.json` pins all image bytes
and both the rendering and evaluation transforms, including the deliberately
incorrect reference. Regenerate with:

```powershell
python "examples/opencv tools/feature_evaluation/build_example.py"
```
