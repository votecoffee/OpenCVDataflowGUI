# Application examples

The [applications index](../README.md#applications) lists saved projects
for part location, conveyor inspection, measurement, classification, and
code reading. Each application folder contains its own project and setup notes.

## Set up the matcher

If the matcher is already installed in the Python environment running
OpenCVDataflowGUI, open a project that uses it directly. DataflowGUI
loads the installed integration when it encounters a geometric matcher block.
For a fresh environment, install the verified
[OpenCVGeometricPatternMatch revision](https://github.com/votecoffee/OpenCVGeometricPatternMatch/tree/632e81bbd12c6640d1c6f65618bd85e45aa26b9e)
from the host repository root. The `--no-deps` option reuses the host's
OpenCV contrib wheel.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps "opencv-geometric-pattern-match @ git+https://github.com/votecoffee/OpenCVGeometricPatternMatch.git@632e81bbd12c6640d1c6f65618bd85e45aa26b9e"
```

## Set up pixel segmentation

See [Pixel segmentation](pixel_segmentation/README.md) for installation and the saved projects.

## Run image classification

Open either [bundled VegQual project](image_classification/README.md) to compare
two training profiles on the same labeled images. **Classify image** uses
DataflowGUI's existing OpenCV and Qt dependencies.
