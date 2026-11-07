# Application examples

The [applications index](../README.md#applications) lists saved projects
for part location, conveyor inspection, measurement, color checks, and
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

## Set up image classification

Install the classifier in the DataflowGUI environment. The `--no-deps` option
keeps the host's existing OpenCV and Qt versions.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --no-deps "opencv-simple-classifier @ git+https://github.com/votecoffee/OpenCVSimpleClassifier.git@1283d62607768264bc60a206c8c2a82863bcaeb9"
```

Open the [VegQual application](https://github.com/votecoffee/OpenCVSimpleClassifier/tree/1283d62607768264bc60a206c8c2a82863bcaeb9/examples/vegqual)
from a checkout of the classifier repository. It includes labeled filmstrips
with held-out prediction review, a trained surface texture model, and a
separate test sequence. To add the tool to a new session
without opening the example, use **File → Load Python plugin…** and select
`integrations/opencv_dataflow_plugin.py` from that checkout.
