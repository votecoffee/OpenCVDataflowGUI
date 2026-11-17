# Batched image inference

Open a project below and run once. No downloads, plugins or learned models are
needed. Three original MIT color captures enter Batch serial entries; the native
ONNX adapter emits source-owned rows. Select group entry and Unpack image
inference result expose the selected source image and its single-image tensors.

| Project | Expected behavior |
| --- | --- |
| [dynamic_tail.json](dynamic_tail.json) | Submit two images and a one-image tail; inspect the green second row. |
| [padded_fixed.json](padded_fixed.json) | Submit two fixed two-row calls, discard one repeated padding result, return three real rows. |
| [source_size.json](source_size.json) | Run at the common original 40×24 size through a dynamic spatial model. |
| [mixed_sizes.json](mixed_sizes.json) | Independently resize three source sizes to a common 8×8 network grid. |
| [grayscale.json](grayscale.json) | Prepare one grayscale channel and classify the selected scalar score as gray. |
| [serial_results.json](serial_results.json) | Unbatch all three results; downstream tools execute three times, ending on blue. |
| [reject_tail.json](reject_tail.json) | Intentional error: three images are not a multiple of two. No inference results. |
| [wrong_signature.json](wrong_signature.json) | Intentional error: the caller declares two-row calls but requests a one-row tail. |
| [mixed_source_sizes.json](mixed_source_sizes.json) | Intentional error: Source mode cannot combine different dimensions in one group. |
| [wrong_output_axis.json](wrong_output_axis.json) | Intentional error: the model collapses the batch axis, so results cannot be assigned to images. |

See Results → Info on the inference block for actual native calls, row counts,
shapes and padding. Each Results group entry retains its own ticket and frame.
The serial example deliberately accepts all three labels; it demonstrates ordered
execution rather than a pass/fail classifier specification. Other successful
examples inspect green row 1, except the one-channel grayscale example.

Images, ONNX arithmetic models, graphs and independent references are original
work by OpenCVDataflowGUI contributors under the repository's MIT license.
On-canvas comments carry this attribution. No customer or third-party assets or
trained parameters are included. The models test numerical channel means/squares,
not general recognition accuracy. `truth.json` records exact model/image hashes
and independent channel arithmetic. Native accumulation tolerance is 2e-6 for
8×8 inputs and 2e-5 for the 40×24 source-size control.

`build_example.py` regenerates the controls with the optional ONNX test package.
It preserves existing different assets; use a fresh folder for regeneration.
See [batch contracts](../../../docs/batch-inference.md) for shape, padding,
selection, failure, memory and provenance limits.
