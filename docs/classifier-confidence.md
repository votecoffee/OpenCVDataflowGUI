# Classifier confidence calibration

Train and evaluate a confidence mapping for a fixed image classifier. The mapping
learns one positive temperature by minimizing multiclass negative log likelihood
on labeled calibration scores, then uses softmax of logits divided by that
temperature. Ordered positive probabilities are converted to log scores before
scaling. Temperature scaling preserves class ranking in exact arithmetic.
[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)

1. Run a fixed classifier on a calibration split. Retain the complete score vector
   in the model's ordered classes. Build one record per image with id,
   capture_group, label, and scores. Use group IDs that identify correlated
   captures, and declare any known training groups.
2. Use **Fit classifier confidence calibration**. Connect its versioned
   calibration output or save it in the **Calibrate confidence and abstain**
   block's calibration parameter for deployment. The fit report includes raw and
   calibrated loss and flags fits at the temperature bounds.
3. For an ONNX classifier, connect both tensors and receipt from the same
   **DNN ONNX inference** block. Expose receipt in the inspector and select
   detailed mode to check the actual tensor hash. When no receipt is available,
   supply the same explicit SHA-256 scorer/preprocessing identity at fit and apply.
   For ONNX, the score identity comes from the model bytes, DNN configuration,
   preprocessing, selected execution settings, and chosen output name.
4. Use **Evaluate classifier reliability** with a separate held-out split.
   Set a null label on deliberately ambiguous or unknown controls. Review how
   many known images and ambiguous controls the gates accepted, along with the
   reliability plot and table.

The blocks verify the declared class order and score kind across fit, application,
and evaluation. The data producer owns the meaning of each tensor column.
Input limits are 2 to 128 distinct classes, up to 4,096 images, and at most
262,144 scores per table. Likelihoods, margins, truncated top-K lists, and vectors
with zero or non-unit probabilities need their own model-specific preparation.
Capture group checks use the supplied IDs. Keep similar frames and repeated images
in the same group and reserve distinct test groups before calibrating.

The apply block accepts a label when the calibrated top probability and its gap
to the second probability both reach their configured minima. An abstention is
an inspection failure with explicit reasons. The candidate, ordered probabilities,
thresholds, and revisions remain in the decision record. The label port emits
nothing on abstention. Choose any final ticket policy separately.

The evaluator compares raw and calibrated negative log likelihood, multiclass
Brier score, and top-label expected calibration error (ECE) on known examples.
ECE weights each equal-width confidence bin's absolute accuracy/confidence gap
by its sample count. Empty bins carry no estimate. The plot shows the mean
confidence and empirical accuracy of each occupied bin. Lean runs can skip the plot.
Coverage is accepted known images divided by all known images. Accepted accuracy
is conditional on that accepted subset and has no value when none were accepted.
The report separately counts accepted ambiguous controls.

The example uses a deliberately overconfident synthetic color score rule. A
temperature fitted on one split can worsen calibration on another. Use separate,
representative captures to validate calibration and rejection gates for the actual
job, especially after changing the model, preprocessing, or acquisition conditions.
Temperature calibration alone provides no unknown-class detection guarantee.

[Runnable examples](<../examples/opencv tools/confidence_calibration/README.md>)
