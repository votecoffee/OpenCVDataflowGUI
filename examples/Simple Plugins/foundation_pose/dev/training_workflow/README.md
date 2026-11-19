# SimpleFoundationPose setup walkthrough

Open **held_out_frame13/run_project.json** or **wrong_object/run_project.json**
from the [FoundationPose examples][examples]. Select the pose block and open
**Set up object…**. Use a project copy when editing. The editor retains CAD,
registered RGB-D, calibration and an instance mask. The isolated worker uses
the same published model weights for each object.

1. Review the CAD XYZ spans and diameter in millimeters. The CAD cloud shows
   the original part frame and origin. The measured reference cloud is in the
   camera frame. Check the source dimensions, focal lengths and depth range.
2. Select **Mask/depth support** in the filmstrip. Turquoise marks selected pixels
   with usable registered depth. Orange marks selected pixels missing depth or
   below 100 mm. Adjust the mask on the source grid and review the counts above
   the filmstrip. **Undo mask** and **Restore imported mask** are available.
3. Use **Runtime on this machine…** to choose [the isolated CPU worker and
   pinned weight folder][worker]. **Check worker and weights** checks the
   worker protocol and the SHA-256 of both ONNX files. Pose preview exercises
   ONNX session creation and inference. Local runtime choices take effect when
   saved in that dialog. A complete draft can be staged before this check.
4. For the held-out project, run the graph once to publish the wired capture and
   proposals, then choose **Preview connected capture** in Setup. For the
   wrong-object project, load its **coarse.json** for preview and choose its
   **capture.npz** under **Preview independent capture…**. The reference
   diagnostics stay visible while the editor shows the new pose overlay, fit
   cloud and ranked candidates. The fit cloud opens with flyaround enabled.
5. **Stage object setup** creates a new recipe and stages its path in the
   inspector. Click **Apply** and save the project. Closing the editor with
   **Cancel** discards its draft.

## Verification

Run the actual inspector offscreen from the OpenCVDataflowGUI checkout. The
checker copies the two graphs and their inputs to a fresh scratch directory.

~~~powershell
$env:PYTHONPATH=(Resolve-Path src).Path
$examples='examples/Simple Plugins/foundation_pose'
& .venv/Scripts/python.exe "$examples/dev/training_workflow/check_training.py" $examples 'C:/scratch/foundationpose_setup_check'
~~~

Both checks used the published ONNX models with supplied masks and two coarse
proposals. Each produced zero accepted poses. The first registration after the
worker and weight check counted as request 1. The candidate data, overlay and
colored cloud matched the equivalent graph execution exactly, and both drafts
survived Stage, Apply, Cancel and byte-for-byte recipe Save As. See the
[replay receipt](qualification.json). These examples cover editor behavior and
rejection review. Acceptance accuracy, global initialization and hands-on
operator review require separate evidence.

## Attribution

The example RGB-D, instance masks and CAD derive from **T-LESS: An RGB-D Dataset
for 6D Pose Estimation of Texture-less Objects**, Tomáš Hodaň, Pavel Haluza,
Štěpán Obdržálek, Jiří Matas, Manolis Lourakis and Xenophon Zabulis, WACV 2017.
The derived captures retain attribution, transforms and preparation details
with their original example documentation. [T-LESS project and license][data] ·
[CC BY 4.0][cc].

[examples]: https://github.com/votecoffee/OpenCVDataflowGUI/tree/main/examples/Simple%20Plugins/foundation_pose
[worker]: https://github.com/votecoffee/OpenCVDataflowGUI/blob/main/docs/simple-pose-runtimes.md
[data]: https://cmp.felk.cvut.cz/t-less/
[cc]: https://creativecommons.org/licenses/by/4.0/
