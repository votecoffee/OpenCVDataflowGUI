# SimpleFoundationPose examples

Open a project below with **File → Open**, then use **Run once**. The app loads
the required Simple Plugin automatically. Each listed folder contains runnable
project JSON files directly; testing and development material is in `dev`.

| Example folder | Start here |
| --- | --- |
| [held_out_frame13](<held_out_frame13/README.md>) | [review_project.json](<held_out_frame13/review_project.json>) |
| [wrong_object](<wrong_object/README.md>) | [review_project.json](<wrong_object/review_project.json>) |

**Inference setup:** select the isolated worker and verified weights once in the
block's setup popup. These machine-local choices persist across projects; see
[pose runtime setup](../../../docs/simple-pose-runtimes.md). Use `run_project.json`
for actual inference or `review_project.json` for file-only recorded results.

Open **Set up object…** to review the CAD and RGB-D reference, inspect mask/depth
support or check the worker and weights. [Setup walkthrough and replay](dev/training_workflow/README.md).

[Detailed setup, sources and limitations](dev/README.md) ·
[Development and verification files](dev/)
