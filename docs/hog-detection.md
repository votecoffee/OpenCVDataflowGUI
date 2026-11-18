# HOG sliding-window detection

`ml.hog_detect` searches an image using OpenCV's native HOG detector. It is a
separate operation from `inspect.hog`, which measures one complete window.
Use the [thirteen projects](<../examples/opencv tools/hog_detection/README.md>)
to try training, detection, selection, persistence and failure controls.

## Models and training

`ml.hog_detector_model` configures either OpenCV's default people detector
(64×128), Daimler people detector (48×96), explicit weights, or a trained binary
linear image SVM. Native preset geometry is fixed. Explicit models expose window,
cell, block, block stride, orientation bins, gamma, signed gradients, Gaussian
sigma, L2-Hys threshold and gray/BGR working encoding. The derivative aperture and
normalization method remain OpenCV's fixed native defaults. Block strides must be
powers of two because the native detector's gradient cache uses that alignment.

For a custom detector, load identically sized uint8 windows, measure `inspect.hog`,
batch descriptors, and use `ml.svm_train` with LINEAR kernel and **no additional
normalization**. Include both positive and negative image classes. Connect the
trained SVM to the model block, select `linear_svm`, and name the positive class.
The converter checks the full HOG feature contract and derives geometry from it;
its explicit-geometry controls do not replace training settings. RBF, multiclass,
normalized or incompatible descriptor models are rejected. Both binary class
orientations are checked against native raw SVM predictions.

Explicit coefficient vectors contain D feature weights and an optional final
bias. A missing bias becomes zero. Portable `opencv_hog_detector_v1` models retain
exact float32 weights, full geometry, class label, source metadata, OpenCV version
and a content revision. Save/load blocks use strict JSON, reject duplicate keys
and nonfinite data, enforce an 8 MiB limit, and verify revision/version. Saving
refuses an existing destination. Saved-project asset sealing and Save As preserve
model bytes; moving a model to another OpenCV version requires requalification.

## Search, grouping and scores

Set `pyramid_levels=1` for a single scale; otherwise `scale_factor` controls the
pyramid. Window stride must be a whole multiple of model block stride. Native
symmetric padding is rounded up to that block stride. Native `none`, `rectangles`
and `meanshift` grouping runs before selection filtering. Rectangle grouping needs
a whole group threshold; mean-shift accepts the native continuous threshold.

`hit_threshold` compares the inclusive native margin, including saved bias. This
is not a probability. Collection items retain `native_weight` and label. Optional
`sigmoid` item scores provide normalized numbers for downstream consumers, with
an explicit uncalibrated score kind. The block's overall score remains unset.

Optional IoU suppression ranks by raw native weight, preserving negative margins
and deterministic coordinate/index ties. Overlap strictly above `nms_iou` is
suppressed; equality survives. It is separate from native grouping. A configured
minimum/maximum retained count produces an ordinary inspection judgment and
`hog.count` fail reason, rather than an execution exception. Maximum zero disables
the count upper limit. Native errors emit no current result and remain recoverable.

## Selection, images and results

Optional wired model/mask/region inputs must produce a current value. A missing
wired value never silently selects a fallback model or the whole image. An unwired
model selects the configured native people preset. Mask and region intersect;
transparent pixels are excluded. Select box center, selected visible-box coverage,
or both. This filters returned detections: gradients still use the entire image.
Native boxes are already clipped to visible image bounds, so coverage is measured
on that visible box and cannot recover an unclipped detector footprint.

Detection uses a fixed uint8 gray/BGR working conversion. uint16 is mapped by its
full numeric range; floats must use [0,1]. Original source pixels, depth, encoding,
ticket and spatial/pyramid payload remain unchanged on the image output. Boxes
and drawable labels use the input image frame; no orientation is measured. When
a smaller pyramid input is selected, its inherited frame metadata identifies that
level for downstream mapping. Graphics obey global/per-block graphics settings.

## Native lifetime and resource controls

Image axes/pixels, padded multiscale windows, descriptor work and estimated
workspace are checked before native search. Candidate count is checked before
quadratic IoU suppression, and retained count has its own limit. Defaults permit
100,000 windows, 2,048 candidates and 512 results. Exceeding a limit rejects the
invocation instead of truncating results. Workspace figures are conservative
estimates, not measured process memory or an allocation guarantee.

HOG searches use a calling-thread CPU scope and restore its previous OpenCL
preference on success or exception. HOG's native SVM upload can still lazily touch
OpenCL. Graph scheduling, desktop workers and live preview therefore prepare the
native context on their caller before short-lived workers start. This extends the
existing KCF startup safeguard and fixes a reproduced Windows OpenCL device-release
abort at worker shutdown. Plugin/client worker integration is described in
[architecture](architecture.md#native-worker-startup). No global thread-count or
process-wide OpenCL setting is changed.

## Qualification and limits

Original synthetic training windows and a separate noise-seeded query scene have
two known 32×32 boxes. At threshold 0.5, native searches over one/three levels and
independent NMS retain both exact boxes. Threshold zero retains six results after
suppression, including four extras. The independent reference explicitly measures
501 windows over three scales and calculates descriptor/weight dot products;
native/independent candidate boxes match and maximum margin error is below
3.3e-8. Training/query hashes differ, but these related controlled shapes are not
an independent industrial accuracy dataset. Thresholds are illustrative.

Native preset tests use blank images and verify geometry/integration, without a
pedestrian accuracy claim. Exact assets, deterministic regeneration, Debug/Lean,
uint16, selection, valid fail/error controls, failure-gated inputs, JSON/XML/Save
As, cold Run All, parallel branches and live preview are checked offscreen. This
qualifies a bounded CPU detector baseline; application accuracy, confidence
calibration, speed benefit and accelerator suitability remain separate.
