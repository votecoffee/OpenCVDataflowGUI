# ONNX model contracts and execution receipts

Single-image, named multi-input, batch and EfficientSAM inference share three controls:

| Field | Behavior |
| --- | --- |
| Contract path | Project-relative `contract.json`, using `/`. Uses its bundled `model.onnx` and checks pinned block settings. Model path is ignored while bound. |
| Receipt mode | Summary (default): identity/settings/shapes. Detailed: tensor hashes too. Off: no receipt output. |
| Fallback policy | Error (default), or opt-in CPU fallback for an unavailable requested target. |

Receipt is a data output hidden as a connector by default. Expose it through the
sidebar to connect a reporting tool; Results also shows the value. It carries the
same ticket/context as inference. Off does not bypass bound-contract checks.

## Export and use a bundle

From the application environment, export a configured block to a **new** folder:

```powershell
python -m opencv_dataflow.assistant_cli export-model program.json --block infer --output model_bundle
```

The response includes the manifest revision and `processing_executed: false`.
Export reads model/configuration files without running the graph, starting
services, training or acquiring images. Existing folders are preserved. Put the
bundle inside the project directory, set Contract path to
`model_bundle/contract.json`, and apply configuration. Python callers use
`opencv_dataflow.model_contracts.export_contract(block, new_directory)`.

The version-1 manifest records exact block type/settings, optional expected
external preprocessing, model byte count/SHA-256 and a canonical content revision.
Source paths and receipt verbosity are excluded from configuration identity;
preprocessing, input declarations, bounds and fallback policy are included.
Each bundle contains `contract.json` and an unchanged `model.onnx`. Save As copies
both through normal project assets, preserving bytes and rejecting conflicts.
JSON/XML projects omit native handles.

`ml.efficientsam` uses the fixed, named EfficientSAM image/prompt preprocessing
profile in its bundle. Its source masks are decoded separately, so candidate,
quality, membership and area thresholds can change without resealing weights.
See [prompted segmentation](prompted-segmentation.md).

For changed settings, explicitly export another folder, then select its contract.
Export can reuse the verified model from a bound bundle; running new settings
against the old revision rejects. Export describes configured values; effective
recipe values must match the active contract during execution.

## Preprocessing and receipts

Image preprocessing now includes a versioned `dnn_image_preprocessing_v1`
descriptor: dimensions, color, layout, resize, padding, scale, mean/std and revision.
Image-to-tensor payloads retain it. Single-image/batch manifests already pin their
preparation settings. Actual dynamic dimensions remain in each call's receipt.

Multi-input preparation happens upstream. Pin those declarations with an expected
native-input-name-to-descriptor mapping, passed as `preprocessing=...` to the export
function or `--preprocessing expected.json` to the CLI. Numeric auxiliaries use
`null`; an empty mapping leaves upstream preparation unpinned. These declarations
do not authenticate arbitrary user code that produces tensors.

`onnx_inference_receipt_v1` contains configuration/revision and records per native
call: loaded model identity, bound contract revision, Python/NumPy/OpenCV versions,
OpenCV build hash, input/output names/shapes/types, preprocessing declarations,
cache reuse, requested/selected execution settings, availability IDs and fallback.
Verification flags distinguish pinned block settings from optional expected
upstream preparation; an unbound receipt does not claim either contract check.
Detailed hashes cover contiguous native tensor bytes with dtype/order stated;
pixels and weights are not embedded. Receipts describe execution, without asserting
recognition accuracy or independent validation of a new image.

Batch receipts retain every call. Each real image also receives an
`onnx_image_execution_receipt_v1` reference with group index, native call/row index
and that call's evidence. Unpack image inference exposes its optional Receipt port.
Padding creates no acquisition/receipt row. Later-chunk errors publish no partial
image results or current receipt events.

## Availability, integrity and scope

Receipts now include `model_import`: the native engine argument, frozen byte
count and import transport. Classic uses an owned temporary snapshot to preserve
explicit native dispatch; Auto/New retain memory import. The file is removed
before forward or after an import exception. Cache/reset and artifact identity
rules are unchanged. See [ONNX engine loading](onnx-engine-loading.md) for the
upstream issue, exact guarantees and the distinction from native placement proof.

CPU requests on the OpenCV/default backend do not enumerate accelerator devices.
Receipts mark availability scope `requested_cpu_only` and list only CPU; this is
not a complete device inventory. Native load/forward still validates the selected
execution path. Explicit accelerator requests retain `backend_discovery`; an
opt-in CPU fallback does not repeat accelerator discovery. This avoids initializing
OpenCL devices on a transient worker merely to run CPU inference, which caused a
native OpenCV shutdown abort on the supported Windows installation.

Opt-in fallback selects OpenCV/classic/CPU, logs a warning and records
`requested_target_unavailable`. Invalid engine/target combinations still reject;
import, binding or forward errors do not silently retry. Native internal
placement/fallback is explicitly **unverified**: selection and availability do not
certify where every layer ran. Native API context:
[OpenCV DNN](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/include/opencv2/dnn/dnn.hpp).

Bound bytes are read and checked once per activation before cache reuse, including
same-size/restored-timestamp changes, then shared across that activation's batch
calls. Limits are 256 MiB per bundled model and 128 KiB per strict UTF-8 manifest.
Unbound paths retain the ordinary stat-based cache; receipts identify loaded bytes,
without certifying unchanged current disk content. Summary omits tensor hashing;
Off omits receipts, while pinned-byte verification still costs I/O. Content hashes
detect changes; they are not signatures or trust certificates.

[Ten controls](<../examples/opencv tools/model_contracts/README.md>) use original MIT
images/models and independent arithmetic. CPU fallback and numerical agreement
are qualified on the pinned build; accelerators and arbitrary exports remain
separate work.
