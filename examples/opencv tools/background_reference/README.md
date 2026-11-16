# Background references across restart

Open a project and press **Run once**. MOG2/KNN should report **Ready** on the
first current image, with an exact foreground mask and a clean background.
No camera, plugin, model download or training environment is needed.

| Project | Expected behavior |
| --- | --- |
| [mog2.json](mog2.json) | Reconstruct MOG2 from the saved clean image, then inspect a part immediately with learning frozen. |
| [knn.json](knn.json) | Restore KNN before each frozen query; persistent parts do not become background. |
| [guarded_mog2.json](guarded_mog2.json) | Permit updates with a region-generated known-foreground mask protecting the part. |
| [guarded_knn.json](guarded_knn.json) | Protect KNN updates by replacing marked samples with trusted background before learning. |
| [replay.json](replay.json) | **Run all** checks five distinct captures: persistent parts, changed positions and an empty last frame. Each mask must match geometric truth exactly. |
| [float_mog2.json](float_mog2.json) | Restore native [0,255] Float32 values, including fractional samples from lossless TIFF. |
| [camera_mismatch.json](camera_mismatch.json) | Intentionally reject a different declared camera ID before native work; no foreground emits. |
| [prepare_reference.json](prepare_reference.json) | Capture the clean image and explicitly write `capture.generated.json` next to the project. Generated files stay ignored. New mode preserves existing files; choose Replace to update one. |

Inspect the background block's **Status** for the reference ID, replay verification,
current activation count and learning policy. The saved model contains a clean
image and binding metadata, not opaque OpenCV state. Current outputs retain their
own tickets and coordinates after reopening a job.

KNN in this wheel does not reliably freeze at a zero native learning rate. Its
frozen-reference workflow rebuilds a disposable query model each activation;
this has a greater processing cost than retained frozen MOG2. Guarded updates
require a correct foreground mask on every activation. Declared camera and
calibration IDs must match, and current grid, depth and encoding are checked.
These checks do not verify physical camera identity or prove that a reference
is clean. See [contracts and limits](../../../docs/background-reference.md).

All nine PNGs, two TIFFs, two sealed reference records, projects and the
[generator](build_example.py) are original OpenCVDataflowGUI fixtures under the
repository's [MIT license](../../../LICENSE). No customer or third-party images
are included. Every project has an attribution comment. Regenerate with:

```powershell
python "examples/opencv tools/background_reference/build_example.py"
```

The [manifest](truth.json) pins all thirteen assets. Automated checks compare
first-image masks to independent pixel/region truth, verify long-lived part
protection and exact regeneration, and run the projects in Debug/Lean modes.
The preparation example is tested in a separate writable copy.
