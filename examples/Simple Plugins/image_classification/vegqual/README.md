# VegQual produce classifier

This saved DataflowGUI project classifies whole crops of individual vegetables
into the publisher's four selected classes. Open [Surface texture](project.json)
or [Color and layout](color_and_layout.json), select
**SimpleClassify**, and click **Train image classifier…** to inspect the
filmstrips and validation report. Use **Run once** to step through the separate
test sequence. The saved model and all example crops are included.

Both recipes used the same split with OpenCV 5. They share the image files
and store their models in separate revision folders.

| Profile | Validation balanced accuracy | Correct test predictions |
| --- | ---: | ---: |
| Color and layout | 70% | 8 / 16 |
| Surface texture | 95% | 10 / 16 |

The validation set has 20 crops. The separate test set has 16 crops. Use
**Misclassified held-out images** in the trainer to inspect the validation
mistake, then step through the test images in the saved graph.

| Publisher label | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| fresh_potato | 24 | 5 | 4 |
| defected_Potato | 24 | 5 | 4 |
| fresh_capsicum | 24 | 5 | 4 |
| defected_capsicum | 24 | 5 | 4 |

Run from the DataflowGUI repository root.

```powershell
opencv-dataflow-run "examples/Simple Plugins/image_classification/vegqual/project.json" --all-images
opencv-dataflow-run "examples/Simple Plugins/image_classification/vegqual/color_and_layout.json" --all-images
```

[Dataset attribution and ground truth](SOURCES.md)
