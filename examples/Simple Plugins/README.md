# Simple Plugins

In a new graph, expand **Simple Plugins** in the toolbox or search for the tool
by name. SimpleClassify is bundled. SimplePatternMatch, SimpleSegmentation, and
SimplePathWidth, SimpleReferenceInspect, and Simple3DLocate load their installed integrations when you add a block or open
a project.

| Tool | Example projects |
| --- | --- |
| **SimpleSegmentation** | [Real ARMBench tote foreground](pixel_segmentation/armbench_tote/README.md), plus [other segmentation projects](pixel_segmentation/README.md). |
| **SimpleClassify** | [VegQual surface texture and color/layout classification](image_classification/README.md) |
| **SimplePatternMatch** | [VisA PCB photographs](pattern_matching/visa_pcb2/README.md) for real board localization and [asymmetric brackets](pattern_matching/asymmetric_brackets/README.md) for generated multi-part scenes. |
| **SimplePathWidth** | [Real industrial weld bead](path_width/weld_bead/README.md), plus a [generated curved bead](path_width/README.md) for controlled gap/width regression. |
| **SimpleReferenceInspect** | [Real clear-tote inspection](reference_inspect/armbench_tote/README.md), [posed PCB hard case](reference_inspect/visa_pcb2/README.md), and [generated reference inspection](reference_inspect/README.md). |
| **Simple3DLocate** | [Real T-LESS clutter with depth verification](3d_locate/real_tless/depth_verified/README.md), plus [generated rigid surfaces](3d_locate/README.md) for controlled regression. |

## Start with real public data

The [real-data showcase](real_data/README.md) links runnable projects for all
six tools, with 28 actual replays, input/result images, numeric outputs and
native 3D fit clouds. It includes dataset licenses, attribution and retained
errors. The new [clear-tote reference inspection](reference_inspect/armbench_tote/README.md)
uses separate empty/loaded captures; the [posed PCB reference example](reference_inspect/visa_pcb2/README.md)
keeps real held-out defect masks and known false alarms visible.

## Set up SimplePatternMatch

If SimplePatternMatch is already installed in the Python environment running
OpenCVDataflowGUI, open a project that uses it directly. DataflowGUI
loads the installed integration when it encounters a SimplePatternMatch block.
From the DataflowGUI repository root, install your local SimplePatternMatch
checkout or wheel that includes the DataflowGUI integration. Replace the
sample path with your checkout or wheel. The `--no-deps` option reuses the
host's OpenCV contrib wheel.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps "C:\path\to\SimplePatternMatch"
```

With **Render overlay** enabled, select **Result / overlay** on the matcher
after a run. The same image appears in the **Image results filmstrip** when
**Capture result thumbnails** is checked. Balanced and Debug retain inspector
image previews.

## Set up SimpleSegmentation

See the [SimpleSegmentation projects](pixel_segmentation/README.md) for the package setup and saved projects.

## Run SimpleClassify

Open either [bundled VegQual project](image_classification/README.md) to compare
two training profiles on the same labeled images. SimpleClassify uses the
app's OpenCV and Qt dependencies.
