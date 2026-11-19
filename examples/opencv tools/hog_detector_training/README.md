# HOG detector training with hard negatives

Start with [train.json](train.json) and choose **Run all**. The graph batches six
positive and six negative crops. Once both batches complete, it mines two
separate target-free images and refits the model with the highest-scoring 20
distinct hard negatives. Connect the trainer's Model port to **Save HOG detector
model** to write your own model.

| Project | Baseline detections | After mining | Known targets |
| --- | ---: | ---: | ---: |
| [heldout_parts.json](heldout_parts.json) | 3 | 2 | 2 |
| [heldout_second.json](heldout_second.json) | 3 | 2 | 2 |
| [heldout_background.json](heldout_background.json) | 1 | 0 | 0 |

The comparison projects load both bundled detector models and use the same
threshold and suppression settings. Both detectors locate all four known
targets in the positive scenes. After adding the hard negatives, none of the
three held-out false positives remain. The mining backgrounds and the held-out
scenes use distinct placements and noise seeds. The report also records the
89 baseline candidates on the mining images and zero after refitting. Those
training-set counts use a zero margin threshold, while the held-out graphs
use 0.5 for both detectors.

All images and models are original synthetic MIT fixtures. truth.json records
the intended target boxes, observed outputs, training details and asset hashes.
To regenerate the printed results and assets, run build_example.py --output PATH
with a new directory. The sampled fixture measures synthetic recall and false
positives. Train and validate on representative captures for production use.

[Training contract](../../../docs/hog-detector-training.md)
