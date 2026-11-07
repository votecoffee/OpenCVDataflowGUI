# Locate asymmetric brackets with SimplePatternMatch

These two projects share one trained geometric recipe and two generated tray
scenes. The sample contains only synthetic imagery.

| Project | Recipe route |
| --- | --- |
| [Saved recipe](project.json) | **SimplePatternMatch** opens a SHA-256-pinned recipe directly. |
| [Wired recipe](wired_recipe.json) | **SimplePatternMatch · Load recipe** sends the same model to **SimplePatternMatch · Match wired recipe**. |

Install the [matcher integration](../../README.md#set-up-simplepatternmatch)
in the DataflowGUI environment. Open either project and click **Run once**
twice. The first scene contains six rotated and scaled brackets. The second
contains four brackets, including one partially occluded bracket, plus a
lookalike. The expected counts are six and four.

Select **Result / overlay** on the matcher or double-click the processed tray
in the image results filmstrip. The overlay shows pose markers, green observed
edges, and red missing edges on the occluded bracket. The **matches** output
contains the accepted poses, coverage, and fit errors. The
[ground truth](expected.json) records the generated datum positions, angles,
and scales in source-image coordinates.

Select the matcher and open **Train geometric recipe…** to see the saved
fine and coarse edges on the training image. To try a new fit, adjust the
training controls and click **Train pattern**. Run the graph to publish
its images. Choose the **Generated tray scenes** output, click
**Capture scene**, and use **Run preview**. To keep a new recipe, click
**Stage recipe**, apply the inspector configuration, and save the project.

From the DataflowGUI repository root, replay either project headlessly.

```powershell
opencv-dataflow-run "examples/Simple Plugins/pattern_matching/asymmetric_brackets/project.json" --all-images
opencv-dataflow-run "examples/Simple Plugins/pattern_matching/asymmetric_brackets/wired_recipe.json" --all-images
```

The [generator](build_example.py) creates both project files, their PNGs, the
pinned recipe, and the ground truth in a new folder. The bracket geometry and
two recipe routes are adapted from the original MIT-licensed
[matcher example](https://github.com/votecoffee/OpenCVGeometricPatternMatch/blob/632e81bbd12c6640d1c6f65618bd85e45aa26b9e/examples/dataflow/build_example.py).

```powershell
python "examples/Simple Plugins/pattern_matching/asymmetric_brackets/build_example.py" path/to/new-example
```
