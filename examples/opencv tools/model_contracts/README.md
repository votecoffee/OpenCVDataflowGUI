# Model contracts and execution receipts

Open a project and run once. No downloads, optional packages or plugins are needed.
Original MIT images/arithmetic models from our batch and multi-input controls are
copied here so the bundles are self-contained.

| Project | Expected result |
| --- | --- |
| [pinned_single.json](pinned_single.json) | Verified model/settings, red classification and summary receipt. |
| [detailed_hashes.json](detailed_hashes.json) | Same output with input/output tensor hashes. |
| [receipts_off.json](receipts_off.json) | Same output/contract checks; no receipt event. |
| [pinned_multi.json](pinned_multi.json) | RGB/grayscale inputs with both upstream preparation descriptors pinned. |
| [pinned_batch.json](pinned_batch.json) | Three source-owned rows, two native calls and per-image receipt references; inspect green row 1. |
| [cpu_fallback.json](cpu_fallback.json) | Requests CUDA; the CPU build explicitly selects CPU and records/logs fallback. |
| [settings_mismatch.json](settings_mismatch.json) | Intentional error: changed scale differs from pinned settings. |
| [checksum_mismatch.json](checksum_mismatch.json) | Intentional error: model bytes reject before native import. |
| [wrong_preprocessing.json](wrong_preprocessing.json) | Intentional error: upstream mean differs from its pinned descriptor. |
| [strict_target.json](strict_target.json) | Intentional error on the CPU build: CUDA unavailable, fallback disabled. |

Receipt is exposed on inference nodes and appears in Results. It identifies model
and contract revisions, preparation, shapes, environment, requested/selected
execution and availability. Native per-layer placement remains unverified. On a
CUDA-capable installation, the two target examples can behave differently; these
are availability controls rather than accelerator benchmarks.

Images, models, graphs, references and on-canvas attribution are original MIT work
by OpenCVDataflowGUI contributors. No learned weights, customer or third-party
data are included. `truth.json` pins model/manifest/image bytes and independent
channel arithmetic. The deliberately changed model must never reach native import.

`build_example.py` regenerates the controls, preserving different existing assets;
use a fresh folder. See [contracts and export API](../../../docs/model-contracts.md)
for new revisions, Save As, receipt options, limits and fallback behavior.
