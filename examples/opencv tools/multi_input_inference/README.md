# Multi-input inference

Open a project and press **Run**. Each image-preparation block shows its original
image and prepared tensor; **Multi-input ONNX inference** shows named outputs,
per-input provenance and the primary image's preprocessing. **Classify** gives a
simple color label so changes to either branch are easy to inspect.

| Project | Expected result |
| --- | --- |
| [paired_images.json](paired_images.json) | RGB and grayscale tensors on independently wired branches; red difference dominates. |
| [two_sources.json](two_sources.json) | Two source grids with matching filename ticket values, explicitly paired across sessions. |
| [reference_input.json](reference_input.json) | A separate reference image is exempt from live-ticket matching. |
| [numeric_aux.json](numeric_aux.json) | A configured gain tensor scales the channels; green becomes dominant. |
| [nhwc.json](nhwc.json) | Explicit NHWC preparation with the corresponding native model. |
| [primary_second.json](primary_second.json) | Slot order differs from model names; slot two owns output image context. |
| [four_inputs.json](four_inputs.json) | Four named tensor inputs contribute to one output. |
| [wrong_shape.json](wrong_shape.json) | Intentional invalid activation; no predictions and a shape diagnostic. |
| [mismatched_tickets.json](mismatched_tickets.json) | Intentional invalid activation; current sources carry different ticket identities. |

No models or images are downloaded. The two negative projects intentionally
return nonzero CLI status while the application remains running.

```powershell
python -m opencv_dataflow.assistant_cli run "examples/opencv tools/multi_input_inference/numeric_aux.json" --summary
```

Try changing the gains in **Create float32 tensor** and rerunning: every activation
rebinds the complete set of model inputs. The role and ticket-policy fields are
dropdowns; input names and expected shapes must match the model. Reference roles
permit cached inputs without disabling recipe consistency. Matching filename
values in two-source replay does not prove exposure synchronization.

Original MIT images, numerical ONNX controls, graphs and annotations by
OpenCVDataflowGUI contributors; no learned parameters, third-party/customer
assets or physical acceptance claims. Each canvas includes attribution. Analytic
scores and asset hashes are in `truth.json`. `build_example.py` regenerates the
collection using optional `onnx`; use a fresh folder to preserve different assets.
See [binding, source, failure and lifecycle contracts](../../../docs/multi-input-inference.md).
