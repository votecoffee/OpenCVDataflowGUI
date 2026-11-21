# CORe50 capture groups

Open [project.json](project.json) in OpenCVDataflowGUI. Select **SimpleClassify ·
Surface texture** and open **Train image classifier…**. The project has four
classes with 24 training and eight held-out images each. The assignments use
specimen IDs for individual objects and recording IDs for capture sessions.

1. Click **Check dataset** to confirm the grouped split. Sessions `s1` and `s5`
   supply training images. All 32 held-out views come from `s7` and use
   different specimens.
2. Open **Capture groups**. Switch between **Recording** and **Specimen** to
   see counts, accuracy and the most frequent wrong label. Select a row and
   click **Show group images** to filter all class filmstrips. Set **Show** to
   **Misclassified held-out images** to narrow the filmstrips further.
3. Close the trainer and click **Run once** twice. The classifier's output
   thumbnail shows the input, ranked scores and confusion. The two images
   show the same unseen plug adapter in different poses. The first predicts
   `plug_adapters` at 30.43%. The second predicts `scissors` at 32.82%.

This model correctly classifies 16 of the 32 held-out views. Use the group
table to see which specimens account for the errors. The two run images
illustrate how recognition can change with pose. [Reference labels,
capture splits, frame hashes and expected scores](reference.json).

## Image attribution

CORe50 RGB dataset subset by Vincenzo Lomonaco and Davide Maltoni, licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The 128 × 128
PNG frames came from the dataset's `core50_128x128` archive. The example
selects and renames frames and derives class labels from the published
specimen categories. The PNG bytes are preserved. [Dataset and license
notice](https://vlomonaco.github.io/core50/) · [CORe50 paper and
citation](https://proceedings.mlr.press/v78/lomonaco17a.html).
