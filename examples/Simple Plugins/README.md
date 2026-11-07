# Simple Plugins

In a new graph, expand **Simple Plugins** in the toolbox or search for the tool
by name. SimpleClassify is bundled. SimplePatternMatch and SimpleSegmentation
load their installed integrations when you add a block or open a project.

| Tool | Example projects |
| --- | --- |
| **SimpleSegmentation** | [Brick mortar, cup rim, and PCB laminate](pixel_segmentation/README.md) |
| **SimpleClassify** | [VegQual surface texture and color/layout classification](image_classification/README.md) |
| **SimplePatternMatch** | [VisA PCB photographs](pattern_matching/visa_pcb2/README.md) for real board localization and [asymmetric brackets](pattern_matching/asymmetric_brackets/README.md) for generated multi-part scenes. |

## Set up SimplePatternMatch

If SimplePatternMatch is already installed in the Python environment running
OpenCVDataflowGUI, open a project that uses it directly. DataflowGUI
loads the installed integration when it encounters a SimplePatternMatch block.
For a fresh environment, install the verified
[OpenCVGeometricPatternMatch revision](https://github.com/votecoffee/OpenCVGeometricPatternMatch/tree/632e81bbd12c6640d1c6f65618bd85e45aa26b9e)
from the host repository root. The `--no-deps` option reuses the host's
OpenCV contrib wheel.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps "opencv-geometric-pattern-match @ git+https://github.com/votecoffee/OpenCVGeometricPatternMatch.git@632e81bbd12c6640d1c6f65618bd85e45aa26b9e"
```

## Set up SimpleSegmentation

See the [SimpleSegmentation projects](pixel_segmentation/README.md) for the package setup and saved projects.

## Run SimpleClassify

Open either [bundled VegQual project](image_classification/README.md) to compare
two training profiles on the same labeled images. SimpleClassify uses the
app's OpenCV and Qt dependencies.
