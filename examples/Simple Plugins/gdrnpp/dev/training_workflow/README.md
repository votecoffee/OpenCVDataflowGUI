# SimpleGDRNPP Setup walkthrough

In OpenCVDataflowGUI, start with the [GDRNPP example list][examples]. Open
**accepted_parts/run_project.json** or **rejected_part/run_project.json** and
select **Set up trained object…** in the SimpleGDRNPP inspector. Use a project
copy for edits. The app loads the pinned setup plugin. To run inference, configure
the [isolated CPU worker and published checkpoint][worker] on this machine.

1. Review the retained RGB image, instance rectangles, capture dimensions and
   focal lengths under **Reference / instances**. Poses map from the named CAD
   frame into the named camera frame in meters. The reference cloud displays
   measured positive finite depth inside the capture mask.
2. Draw or edit rectangles in source-image pixels. The right and bottom bounds
   are exclusive. Select a row to see usable registered depth inside that
   rectangle. Box depth helps check the input. Pose support is evaluated
   separately against the predicted CAD surface during preview.
3. Click **Check checkpoint** to verify and load the checkpoint from the pinned
   recipe in the isolated worker. A newly imported or unpinned setup can be
   previewed and staged. Reopen it after Apply to verify the pinned archive.
4. Click **Preview reference**, **Preview another capture…**, or **Preview
   connected**. Inspect accepted, rejected or overlap-suppressed candidates in
   **Pose / depth review**. Use **Flyaround result…** for the colored metric
   evidence. For a separate capture, open the rejected case and preview
   **live_review/capture.json**.
5. Click **Stage setup** to save a new recipe and stage the inspector fields.
   Click **Apply** in the inspector, then save the project. Reopening restores
   the edited recipe. **Close / discard** leaves the inspector fields as they
   were before opening the dialog. Runtime and model defaults are machine-local
   choices and take effect when selected.

## Verification

Run from the checkout of OpenCVDataflowGUI with its dependencies and the isolated
worker configured. The checker copies each project and sealed recipe to the
specified fresh scratch folder and exercises the actual inspector offscreen.

~~~powershell
$env:PYTHONPATH=(Resolve-Path src).Path
$examples='examples/Simple Plugins/gdrnpp'
& .venv/Scripts/python.exe "$examples/dev/training_workflow/check_training.py" $examples 'C:/scratch/gdrnpp_setup_check'
~~~

The [replay receipt](qualification.json) records one accepted pose from five
proposals and zero accepted from one on the retained examples. The separate
capture used the rejected example's draft settings and yielded zero accepted
from five. Both retained cases loaded the checkpoint without a forward pass,
reused it on the first preview, verified native metric cloud colors, and passed
Stage, Apply, Cancel and exact recipe Save As. These are setup and inference
checks for the same published object model. New-object training, wider accuracy
studies, physical validation and hands-on editor review remain open.

## Attribution

The example captures, supplied visible boxes and CAD are from **T-LESS: An
RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**, Tomáš Hodaň,
Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis and Xenophon
Zabulis, WACV 2017. The dataset is [CC BY 4.0][cc]. The project folders retain
source credit and import details for the RGB-D frames, CAD representation and
reordered instance boxes. [T-LESS project and license][data].

[examples]: https://github.com/votecoffee/OpenCVDataflowGUI/tree/main/examples/Simple%20Plugins/gdrnpp
[worker]: https://github.com/votecoffee/OpenCVDataflowGUI/blob/main/docs/simple-pose-runtimes.md
[data]: https://cmp.felk.cvut.cz/t-less/
[cc]: https://creativecommons.org/licenses/by/4.0/
