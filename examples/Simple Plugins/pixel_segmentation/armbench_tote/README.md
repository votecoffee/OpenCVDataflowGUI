# Segment inventory in a real ARMBench tote

This example uses **SimpleSegmentation** to separate visible inventory from
a blue tote. It trains on two photos and replays a separate loaded tote
and an empty tote. Touching items can form one connected foreground region.

Install the [SimpleSegmentation integration](../README.md#run-in-dataflowgui),
open [the project](project.json), and click **Run once** twice. Select
**Show outlines on the photo** for the foreground outlines on the original
color image. The segmentation block also exposes the binary **mask**, raw
**score map**, and **inspected** region.

This project was verified with [OpenCVDefectSegmentation revision 241b593](https://github.com/votecoffee/OpenCVDefectSegmentation/tree/241b593a51420f3ee78fab85e86e7f6b2bde2b46).

The **Measure blue dominance** block computes the grayscale input used
by the pixel model. The fixed region limits evaluation to the tote cavity.
This example is scoped to the blue tote and similar camera placement.

## Validation

The saved recipe uses a 0.50 vote cutoff and a minimum foreground area of
180 pixels at 612 × 512. These results compare with approximate manual
polygons drawn for this example, within the observed region.

| Held-out photo | Result |
| --- | --- |
| Loaded tote | 94.8% foreground IoU, 98.5% precision, and 96.2% recall |
| Empty tote | Zero foreground pixels after area filtering, over 183,848 inspected pixels |

## Retrain or replay

Open **Train segmentation candidate…** on the segmentation block and load
[`training.json`](training.json). The label files are editable in the trainer.
[`validation.json`](validation.json) points to the separate evaluation labels.
Label values are 0 for ignore, 1 for background, and 2 for inventory. Training
labels leave a narrow ignore band along object boundaries.

```powershell
opencv-dataflow-run "examples/Simple Plugins/pixel_segmentation/armbench_tote/project.json" --all-images
```

The [builder](build_example.py) recreates the color images, model inputs,
masks, and pinned recipe from Amazon's tiny archive and the
[saved polygons](polygons.json). [Source archive, changes, and attribution](SOURCES.md)
