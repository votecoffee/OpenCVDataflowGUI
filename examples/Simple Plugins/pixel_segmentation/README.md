# SimpleSegmentation examples

These examples use **SimpleSegmentation** to learn a foreground class from labels. Each includes a saved project, editable labels, and a pinned model.

| Project | Foreground | What to look for |
| --- | --- | --- |
| [Brick mortar](brick_mortar/project.json) | Bright mortar joints | Labels cover selected joint and brick patches in the upper-left area. The result follows joints across the photograph. |
| [Cup rim](cup_rim/project.json) | Porcelain lip | A coarse rectangular search region keeps the model focused on the cup. The learned mask traces the rim within it. |
| [PCB laminate](pcb_laminate/project.json) | Exposed blue board surface | Run twice to locate two different VisA boards. Training labels come from the first photo. Dark chips and metal pins are mostly background. |
| [ARMBench tote](armbench_tote/README.md) | Inventory inside a blue tote | Train on two real photos, then replay separate loaded and empty totes with optional 2D pick selection. |

## Run in DataflowGUI

Install a current SimpleSegmentation build from a local source checkout or wheel in the Python environment running DataflowGUI. From the source checkout, run that environment's Python with `-m pip install --no-deps -e .`. DataflowGUI loads the installed integration when you open a saved project that uses it.

Open a project above and press **Run once**. Inspect **Overlay** for the highlighted foreground, **Mask** for the binary result, or **Score map** for the raw forest votes. Use **Inspected** to identify which pixels were evaluated. The block's **Threshold** and **Min area px** change the binary output without retraining. Reset flow state to replay a finished sequence.

The examples use four voting workers at native sampling. Adjust **Workers** in the inspector to budget CPU alongside other graph branches. Use one worker to measure the serial block time.

The same voting limit applies at supersampled resolutions and when a batch contains one registered patch. Batches with several registered parts use the workers across the parts.

Compact inspection regions such as the cup example limit feature computation, normalization, and connected-component measurement to the selected window and any needed filter support. The output mask and scores use the source image coordinates. The cached cutoff preview uses a direct threshold at **Min px** of one.

In the PCB project, **Locate one board** takes the external contour of the laminate mask and checks for one board-sized object. Its **Collection** output includes the contour, bounding box, and centroid in the 650 × 495 image grid. **Show board outline and center** draws the contour and centroid over the learned mask. The contour follows visible laminate, so it can curve inward near edge-mounted components. The raw laminate mask and the filled contour mask are separate outputs.

From the DataflowGUI repository root, you can replay both PCB photographs in the same environment:

```powershell
opencv-dataflow-run "examples/Simple Plugins/pixel_segmentation/pcb_laminate/project.json" --all-images
```

## Review or retrain

Select the segmentation block and open **Train segmentation candidate…**. Choose the `training.json` alongside the project and open **Open training labels**. Use **Foreground**, **Background**, and **Ignore** to edit the labels. Save the mask before training. Stage the candidate, apply the inspector configuration, and save the project to use the new model.

The cup and brick results show coverage on images used for training. The PCB example also includes a held-out [validation manifest](pcb_laminate/validation.json) for board 0621. It marks 21,065 pixels (6.5% of the image) across visible laminate, chips, pins, and the background. Unmarked pixels are ignored by the metrics.

In the trainer, select the PCB **Validation manifest**, choose **Open validation labels**, and turn on **Show prediction**. **Compare labels** shows hits in green, missed foreground in red, foreground on labeled background in amber, and predictions on unlabeled areas in purple. Change **Cutoff** or **Min px** to update the current image's overlay and metrics from its cached scores.

Choose **Keep preview settings** while comparing the saved labels. Close the review and choose **Stage settings** for the pinned model or **Stage candidate** for a newly trained model. Use **Apply configuration** in the inspector and save the project. The aggregate training summary lists the cutoff and area used for that report.

Choose **Evaluate validation set** before staging to compare the inspector values with the kept settings on every saved validation image using the same model scores for both rows. The trainer shows the number of observed labeled pixels and precision, recall, and specificity. Select another validation manifest to evaluate the same candidate on a different set.

Results for the bundled PCB model at a minimum area of 70 pixels:

| Cutoff | Precision | Recall |
| --- | ---: | ---: |
| 0.35 | 94.64% | 99.99% |
| 0.55 (saved project) | 95.83% | 99.97% |
| 0.90 | 97.65% | 99.50% |

The figures describe the marked pixels on board 0621. To reproduce the saved cutoff result from the DataflowGUI repository root:

```powershell
ocv-segment evaluate "examples/Simple Plugins/pixel_segmentation/pcb_laminate/model.yml" "examples/Simple Plugins/pixel_segmentation/pcb_laminate/validation.json" --threshold 0.55 --min-area 70
```

## Image credits and preprocessing

- **Brick** reuses the unchanged scikit-image `brick.png`. The original texture is from CC0Textures, distributed under CC0 1.0. [Source and credit](../../assets/photograph-sources.md)
- **Cup** uses an OpenCV BGR-to-grayscale conversion of Rachel Michetti's coffee photograph, courtesy of Pikolo Espresso Bar, distributed under CC0 1.0. [Source and credit](../../assets/photograph-sources.md)
- **PCB** uses VisA PCB2 normal photographs 0000 and 0621 by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer, distributed under CC BY 4.0. The inputs were resized to 650 × 495 pixels with area interpolation and converted to a grayscale blue-excess channel using saturating `B - R`. [Source and credit](../../assets/photograph-sources.md#visual-anomaly-visa-pcb-photographs)
