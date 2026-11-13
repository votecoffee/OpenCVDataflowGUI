# Guided application demos

These projects introduce the editor and runtime with bundled synthetic inputs.
They run without a camera, controller, optional model download or user plugin.
Start the application, choose **File → Open…**, and open a project below.

| Order | Demo | What to learn |
| --- | --- | --- |
| 1 | [Quick-start graph](demo/README.md) | Run once, select blocks, trace wires and inspect images/values. |
| 2 | [Build an inspection](starter_inspection/README.md) | Follow a small program and preview parameter changes using retained inputs. |
| 3 | [Editable regions](regions/README.md) | Draw/select a region and reuse it across inspection branches. |
| 4 | [Object collections](object_collections/README.md) | Inspect data outputs and use collection filters, sorting and overlays. |
| 5 | [Result graphics](overlay_result_graphics/README.md) | Compose selected graphic layers on the source image. |
| 6 | [Image-set workspace](image_set_workspace/README.md) | Run image sets, move node groups and inspect ticketed filmstrip results. |
| 7 | [Validation sets](validation_images/README.md) | Compare an inspection with expected pass/fail criteria and review a report. |
| 8 | [Sequence memory](sequence_memory/README.md) | Observe warmup and state across repeated runs. |
| 9 | [3D data and viewer](rgbd_cloud/README.md) | Inspect typed point-cloud connections and open the native 3D viewer. |

Use **Workspace → Auto arrange blocks** to organize a graph. Save it, move nodes
and save again to retain the layout. In the inspector, **Run block** uses the
last eligible input for one execution; **Live results** previews eligible edits.
Click **Apply configuration** to keep an edited configuration. Double-click a
result image to inspect it in a larger view.

**Run all** processes a finite set once; **Continuous** loops until stopped.
The image-set and validation demos intentionally include failed inspections so
you can see failure reasons and verdict colors. These are useful result states,
not evidence of a broken demo. Sequence tools can initially withhold outputs
while their configured warmup completes.

For an individual vision algorithm, browse [OpenCV tools](<../opencv tools/README.md>),
[scikit-image tools](<../skimage tools/README.md>) or [Custom tools](<../custom tools/README.md>).
For pipeline scheduling, see [Control flow tools](<../README.md#control-flow-tools>).
The Lena function examples are under OpenCV tools and retain their separate
explicit image-preparation step.

[All examples](../README.md)
