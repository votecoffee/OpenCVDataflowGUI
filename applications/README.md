# Geometric inspection applications

These projects use the pinned **Geometric inspection recipe** block. Each folder
contains a saved project, a trained recipe, a reference image, and ordered
16-bit synthetic scenes. The expected files list the accepted match count
and the intended datum locations for each scene.

| Application | Scene order and expected matches |
| --- | --- |
| [Crowded tray grid](tray_grid/README.md) | Four full trays with **42, 42, 42, 42** |
| [Needle row](needle_row/README.md) | Loaded, partially loaded, and empty with **19, 16, 0** |
| [Weighted block geometry](weighted_blocks/README.md) | One true part, two true parts, and look-alikes only with **1, 2, 0** |

## Set up the matcher

Use the Python environment that runs OpenCVDataflowGUI. Clone
[OpenCVGeometricPatternMatch](https://github.com/votecoffee/OpenCVGeometricPatternMatch)
as a sibling checkout if needed, then select the verified revision. The
`--no-deps` install reuses the host's OpenCV contrib wheel.

```powershell
git clone https://github.com/votecoffee/OpenCVGeometricPatternMatch.git ../OpenCVGeometricPatternMatch
git -C ../OpenCVGeometricPatternMatch checkout --detach 632e81bbd12c6640d1c6f65618bd85e45aa26b9e
.\.venv\Scripts\python.exe -m pip install --no-deps -e ../OpenCVGeometricPatternMatch
```

In the GUI, use **File → Load Python plugin…** to load
`integrations/opencv_dataflow_plugin.py` from the matcher checkout before
opening a project. Load it explicitly each session. Run **Continuous** to
replay all scenes, or step through them with **Run once**. Select the
matcher to inspect datum axes, found and missing edges, and other fit
diagnostics. **Train geometric recipe…** opens the trainer. The included
reference is available for training experiments, while replay uses the
existing hash-pinned recipe.

For headless replay from the host repository, pass the plugin explicitly.
Change the project path to replay a different application.

```powershell
$plugin = Resolve-Path ../OpenCVGeometricPatternMatch/integrations/opencv_dataflow_plugin.py
.\.venv\Scripts\python.exe -m opencv_dataflow.runner applications/tray_grid/inspection.json --plugin $plugin --all-images
```

The needle and weighted examples include an intended failed inspection.
Their full replays return exit code 1 after processing all frames. Use
`--report` with a new file path to see per-frame matcher outputs. Copy an
application's entire folder to relocate it with its assets.

## Regenerate the fixtures

The [generator](build_geometric_examples.py) builds all three applications
from deterministic drawn geometry and a fresh training run. Use a new
destination so published assets remain available for comparison.

```powershell
$destination = Join-Path ([IO.Path]::GetTempPath()) ('ogpm-applications-' + [guid]::NewGuid())
.\.venv\Scripts\python.exe applications/build_geometric_examples.py ../OpenCVGeometricPatternMatch $destination
```
