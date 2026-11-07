# SimpleClassify examples

**SimpleClassify** is bundled in the DataflowGUI toolbox. Search for it by
name, or open one of the two saved projects. Both projects share the same
132 labeled VegQual images and a 16-image test sequence.

| Project | Recipe |
| --- | --- |
| [Surface texture](vegqual/project.json) | Classifies using local contrast and texture from nearby training examples. |
| [Color and layout](vegqual/color_and_layout.json) | Classifies using a random forest on global and coarse regional features. |

Click **Run once** to step through the test sequence. The block outputs the
winning label, score, confusion, and the ranked scores for all classes. Scores
are uncalibrated percentages totaling 100%. Confusion is the runner-up score
divided by the winning score, expressed as a percentage.

Select **SimpleClassify** and click **Train image classifier…** to review
labeled filmstrips, held-out predictions, and the confusion matrix. Click
**New class…** and **Add images…** to label files from disk.

To label images from the pipeline, select a class tab and open **Capture from
graph**. Run the graph with the trainer open, then click **Capture image**.
**Wired input / image** captures the latest image on the classifier's input
at the configured pyramid level. The picker also offers the last processed
input and other cached image outputs. Each capture appears in the active
filmstrip. Use **Toggle validation** to hold out selected captures.

After training, stage the model, apply the inspector configuration, and save
the project. Staging copies the captured PNGs into the model revision, so the
filmstrips can reopen with the project.

The [optional plugin entry point](classifier_plugin.py) can also be loaded
through **File → Load Python plugin…**. The bundled implementation was
imported from [OpenCVSimpleClassifier 0.2.0](https://github.com/votecoffee/OpenCVSimpleClassifier/tree/1283d62607768264bc60a206c8c2a82863bcaeb9)
under the repository's MIT license.

[VegQual results and commands](vegqual/README.md) · [Image and label attribution](vegqual/SOURCES.md)
