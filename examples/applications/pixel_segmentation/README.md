# Pixel segmentation

These examples use **Segment image** to learn a foreground class from painted labels. Each includes a saved project, editable labels, and a pinned model. The images come from the [bundled photograph collection](../../assets/photograph-sources.md).

| Project | Foreground | What to look for |
| --- | --- | --- |
| [Brick mortar](brick_mortar/project.json) | Bright mortar joints | Labels cover selected joint and brick patches in the upper-left area. The result follows joints across the photograph. |
| [Cup rim](cup_rim/project.json) | Porcelain lip | A coarse rectangular search region keeps the model focused on the cup. The learned mask traces the rim within it. |
| [PCB laminate](pcb_laminate/project.json) | Exposed blue board surface | Run twice to compare two different VisA boards. Training labels come from the first photo. Dark chips and metal pins are mostly background. |

## Run in DataflowGUI

Install the OpenCV Pixel Segmentation package from its source checkout in the Python environment running DataflowGUI. From the segmentation checkout, run that environment's Python with `-m pip install --no-deps -e .`. Use **File → Load Python plugin…** and select `integrations/opencv_dataflow_plugin.py` from the segmentation checkout.

Open a project above and press **Run once**. Inspect **Overlay** for the highlighted foreground, **Mask** for the binary result, or **Score map** for the raw forest votes. Use **Inspected** to identify which pixels were evaluated. The block's **Threshold** and **Min area px** change the binary output without retraining. Reset flow state to replay a finished sequence.

## Review or retrain

Select the segmentation block and open **Train segmentation candidate…**. Choose the `training.json` alongside the project and open **Open training labels**. Use **Foreground**, **Background**, and **Ignore** to edit the labels. Save the mask before training. Stage the candidate, apply the inspector configuration, and save the project to use the new model.

The cup and brick results show coverage on images used for training. The second PCB photograph is an unlabeled replay image.

## Image credits and preprocessing

- **Brick** reuses the unchanged scikit-image `brick.png`. The original texture is from CC0Textures, distributed under CC0 1.0. [Source and credit](../../assets/photograph-sources.md)
- **Cup** uses an OpenCV BGR-to-grayscale conversion of Rachel Michetti's coffee photograph, courtesy of Pikolo Espresso Bar, distributed under CC0 1.0. [Source and credit](../../assets/photograph-sources.md)
- **PCB** uses VisA PCB2 normal photographs 0000 and 0621 by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer, distributed under CC BY 4.0. The inputs were resized to 650 × 495 pixels with area interpolation and converted to a grayscale blue-excess channel using saturating `B - R`. [Source and credit](../../assets/photograph-sources.md#visual-anomaly-visa-pcb-photographs)
