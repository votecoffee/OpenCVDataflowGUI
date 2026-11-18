# SimpleClassify examples

**SimpleClassify** is bundled in the DataflowGUI toolbox. Search for it by
name, or open one of the two saved projects. Both projects share the same
132 labeled VegQual images and a 16-image test sequence.

| Project | Recipe |
| --- | --- |
| [Surface texture](../vegqual/project.json) | Classifies using local contrast and texture from nearby training examples. |
| [Color and layout](../vegqual/color_and_layout.json) | Classifies using a random forest on global and coarse regional features. |

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
through **File → Load Python plugin…**.

[VegQual results and commands](../vegqual/README.md) · [Image and label attribution](../vegqual/SOURCES.md)

## Result and sample review

SimpleClassify now selects **Overlay** in the result filmstrip: the input
image, ranked class scores, winning score and confusion appear together.
Set **Render overlay** off to skip chart generation; numeric outputs stay
the same. The chart has its own display coordinates, not the source grid.
Scores sum to 100% but are not calibrated confidence. Full labels and all
classes remain in **Probabilities**; the chart shows up to 20 rows.

In training, double-click a filmstrip image or choose **Review selected…**
to zoom/pan the original alongside its scores. **100%** shows source pixels.
Editing the assignments clears the preview model until training completes.

The VegQual [correct](../vegqual/result_views/correct.png) and
[incorrect](../vegqual/result_views/incorrect.png) examples use unchanged
held-out predictions. The latter is a fresh potato predicted as defective;
the new chart does not fix that error. Exact scores and source filenames
are in [predictions.json](../vegqual/result_views/predictions.json).
These derived views retain the [VegQual attribution](../vegqual/SOURCES.md).

Regenerate these previews with the host Python: `python "examples/Simple Plugins/dev/build_result_views.py"`.

## Completed-runtime Results review

The [0.3.2 pinned-plugin examples](completed_runtime/README.md) add an independent
**Review completed classification…** Results action: exact consumed input/model
bytes, full ranked candidate table, winner/runner-up gap, confusion, zoom/pan and
frozen array/model/JSON/PNG exports. Load that folder's **plugin.py first**, then
open its projects; restart if another classifier version is already imported.
No environment installation is needed. The explicitly selected plugin replaces
only the known bundled classifier for this session.

Three projects replay 33 real labeled VegQual frames, with nine frozen exports.
Both existing recipes and test splits are unchanged; their incorrect predictions
remain visible. Scores and default charts exactly match the earlier implementation.
Training remains independent, and Save As carries the model and labeled images.

## Dataset checks and correction

The [0.3.3 pinned-plugin training examples](training_workflow/README.md) add
background checks for unreadable files, exact pixel duplicates, conflicting
class labels and duplicate leakage across training/validation. Planned class and
split counts stay visible; selecting a problem focuses its filmstrip entry for
correction. Checks become stale after edits. They do not train or predict.

Load that folder's **plugin.py first**, then open either original or retrained
VegQual profile. Restart if another classifier version is already imported.
This explicit example selects the standalone disk-image trainer for this session;
it does not include the built-in trainer's **Capture from graph** extension.
The built-in trainer and the host environment have not been modified. Preserve
graph capture when integrating the new checks into the built-in trainer.

Both native workflows pass audit, correction, retraining, Stage/Apply, reopen and
exact model/image Save As. The original 33 runtime frames retain exact scores
and charts. See the [license and attribution](training_workflow/SOURCES.md).
