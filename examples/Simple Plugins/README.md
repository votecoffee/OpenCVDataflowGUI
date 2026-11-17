# Simple Plugins

In a new graph, expand **Simple Plugins** in the toolbox or search for the tool
by name. SimpleClassify is bundled. SimplePatternMatch, SimpleSegmentation, and
SimplePathWidth, SimpleReferenceInspect, Simple3DLocate, Simple3DMeasure and
SimpleBinPicking load their installed integrations when you add a block or open
a project.
SimpleGDRNPP uses explicit plugin loading and a separately configured worker.

| Tool | Example projects |
| --- | --- |
| **SimpleSegmentation** | [Real ARMBench tote foreground](pixel_segmentation/armbench_tote/README.md), plus [other segmentation projects](pixel_segmentation/README.md). |
| **SimpleClassify** | [VegQual surface texture and color/layout classification](image_classification/README.md) |
| **SimplePatternMatch** | [VisA PCB photographs](pattern_matching/visa_pcb2/README.md) for real board localization and [asymmetric brackets](pattern_matching/asymmetric_brackets/README.md) for generated multi-part scenes. |
| **SimplePathWidth** | [Real industrial weld bead](path_width/weld_bead/README.md), plus a [generated curved bead](path_width/README.md) for controlled gap/width regression. |
| **SimpleReferenceInspect** | [Real clear-tote inspection](reference_inspect/armbench_tote/README.md), [posed PCB hard case](reference_inspect/visa_pcb2/README.md), and [generated reference inspection](reference_inspect/README.md). |
| **Simple3DLocate** | [Real T-LESS clutter with depth verification](3d_locate/real_tless/depth_verified/README.md), plus [generated rigid surfaces](3d_locate/README.md) for controlled regression. |
| **Simple3DMeasure** | [Real T-LESS measurement review and generated known/raised/missing surfaces](3d_measure/README.md), with native RGB flyaround, retained reference recipes and explicit dimensional qualification limits. |
| **SimpleBinPicking** | [Suction contact/clearance review](bin_picking/README.md), with five generated controls and three real T-LESS replays; real cases retain zero eligible picks and open grasp qualification. |
| **SimpleGDRNPP (experimental)** | [T-LESS pose reviews and native setup](gdrnpp/README.md), with three retained rejected poses, RGB overlays and native fit-cloud flyaround. |

The [SimpleLaserProfile trainer and examples](laser_profile/README.md) add
retained reference/ROI teaching, six execution projects and eight saved image/
profile/cloud reviews. Load its bundled-wheel plugin explicitly; no environment
installation is needed. Calibrated results use native fly-around clouds. Real
CC BY 4.0 WLI-Set captures remain pixel-only, with poor fits and missing samples
visible. The trainer restores existing training and stages edits for Apply.

The [SimpleSurfaceEnhance setup and execution examples](surface_enhance/README.md)
add retained light/reference/ROI teaching, independent previews, Stage/Apply
and one cached execution plugin. Five rendered controls retain normal/albedo,
relighting, residual and unavailable-reason views, with attributed CC0 material
inputs. Load its bundled-wheel plugin explicitly; file-only map reviews remain
available separately. Acquired-camera accuracy remains unqualified.

## Start with real public data

The [real-data showcase](real_data/README.md) links runnable projects for all
six tools, with 28 actual replays, input/result images, numeric outputs and
native 3D fit clouds. It includes dataset licenses, attribution and retained
errors. The new [clear-tote reference inspection](reference_inspect/armbench_tote/README.md)
uses separate empty/loaded captures; the [posed PCB reference example](reference_inspect/visa_pcb2/README.md)
keeps real held-out defect masks and known false alarms visible.

The newer [Simple3DMeasure examples](3d_measure/README.md) add three real
first-return captures with supplied annotation fixtures. They demonstrate the
measurement/review workflow; independent physical accuracy remains unqualified.

The experimental [FoundationPose saved-result reviews](foundation_pose/README.md)
add source-aligned overlays and native RGB flyaround clouds from two actual
CPU ONNX runs. Both retain rejected results. These fast file-only projects do
not run neural inference or require its worker environment; live integration
and the training editor remain future work.

The [SimpleGDRNPP examples](gdrnpp/README.md) provide three file-only saved-result
reviews without running neural inference. Setup and inference use the explicitly
loaded plugin and an isolated worker; model weights stay external. The native
popup retains RGB-D/CAD references, previews connected captures and stages draft
changes for **Apply configuration**. All retained poses fail the measured-depth
gates. The checkpoint covers T-LESS object 6; new-object training and broader
pose accuracy remain open.

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
