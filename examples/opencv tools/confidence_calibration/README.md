# Classifier confidence and abstention

These projects demonstrate three ways to use the confidence calibration blocks.

| Project | Expected outcome |
| --- | --- |
| [reliability.json](reliability.json) | Fit on 48 calibration records and compare reliability on 40 separate labeled records plus four ambiguous controls. |
| [known_red.json](known_red.json) | Run an actual OpenCV ONNX forward pass. The fitted mapping accepts the red image and emits a red label. |
| [ambiguous.json](ambiguous.json) | Run the same model and mapping. The confidence and margin gates abstain with failure reasons and withhold the label. |

On the supplied split, the temperature is about 1.556. Held-out ECE falls from
0.129 to 0.060 and negative log loss falls from 0.586 to 0.484. The example gates
accept 20 of 40 known controls, with 18 correct among the accepted controls, and
abstain on all four ambiguous controls. The groups, colors, and label noise are
constructed to show calibration and the accuracy/coverage tradeoff.

The two input PNGs, the untrained ONNX mean-color scoring rule, and the score/label
records are original synthetic MIT fixtures from build_example.py. The builder
generates every calibration and test image in memory and runs native OpenCV to
obtain the scores. It requires the optional ONNX Python package to regenerate
the model. Runtime replay uses the included model and the app's usual dependencies.
Keep the project, PNG, and ONNX files together when copying the examples.

[Workflow and score contract](../../../docs/classifier-confidence.md)
