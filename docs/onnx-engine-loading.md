# ONNX engine loading

Single-image, named multi-input and batch inference share a loader that preserves
the engine argument and the verified model-byte snapshot. **Classic** now uses
the explicit native path overload; **Auto** and **New** continue to load from memory.
The engine choices and saved-project schema are unchanged.

## Why Classic has a separate route

OpenCV 5.0.0's [vector-buffer classic branch](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/onnx/onnx_importer.cpp#L3934)
delegates without forwarding its engine argument. That can select the default
automatic route, accepting a model that the actual classic importer rejects.
The [path overload](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/onnx/onnx_importer.cpp#L3867)
preserves the explicit engine argument. The generic `readNet` memory overload
delegates to the same vector route, so changing only the function name does not help.

The application first reads/hashes the model using the existing contract/cache
rules. For Classic it writes those same frozen bytes to a unique owned temporary
`.onnx` file, closes the writer, imports with the explicit classic argument and
removes the temporary file immediately. Inference uses the loaded native weights
after deletion. Native import failures also clean up that file and produce the
ordinary block exception/diagnostic with no current tensor/receipt events.
There is no retry with New after a classic import failure.

The loader never reopens the original model pathname after hashing. If that source
changes between hashing and native import, the current activation still consumes
the verified snapshot; a later activation follows the normal file/revision checks.
Every import has its own file; concurrent blocks do not change global environment
variables to force an engine. Auto/New need no temporary file. ORT retains the
existing native memory behavior and is not qualified by this change.

## Cache, receipts and persistence

The native network and import evidence are reused while the existing cache key
matches. The temporary file is created on import, rather than every cached forward.
Changing engine/settings or the model key triggers a new import. Reset, close and
import/forward exceptions discard the native handle and import evidence.

Each non-Off execution receipt adds a portable `model_import` object:

```json
{
  "model": "onnx_import_v1",
  "transport": "temporary_snapshot_file",
  "engine_argument": 1,
  "snapshot_bytes": 210,
  "temporary_file_retained": false,
  "native_engine_observed": false
}
```

`snapshot_bytes` is the actual byte count; the example above is illustrative.
The parent receipt already records model SHA-256 and requested/selected settings.
Transport is `temporary_snapshot_file` for Classic and `memory_buffer` for the
other current routes. Neither the temporary pathname nor model bytes are added to
the receipt. Runtime import evidence/native handles are not saved as project
configuration; JSON/XML and model-bundle/Save As behavior remain compatible.

The Python `Net` interface exposes no native-engine getter here. Tests distinguish
native classic rejection from New acceptance, and verify the API argument and
frozen bytes. This does not observe every internal layer or certify accelerator
placement. Native process-wide configuration such as `OPENCV_FORCE_DNN_ENGINE`
can supersede the argument; the application does not change that variable. Keep
the existing [model-contract receipt](model-contracts.md) verification limits.

CPU-only inference still avoids unrequested accelerator discovery. A cold-process
classic validation-worker check verifies clean shutdown without querying GPU
availability. Accelerator/ORT qualification and external-data ONNX bundles remain
separate; these fixtures use single-file embedded-weight models.

## Existing examples and qualification

Two original public control models use shared-constant heads rejected by the
classic importer on this build. Their **21** detection/rotated-detection projects
now explicitly select New. Migration changes only the engine value, preserving
model/image hashes, connections, layout, exposure and all other saved fields.
Their generation scripts make the same choice. The optional pinned YOLOX coffee
example retains Classic and is checked through actual classic import/forward.

An existing user project configured as Classic may therefore reveal a previously
hidden compatibility error. Choose New/Auto when appropriate; a bound model
contract must be re-exported if its engine setting changes. User/private projects
are not silently edited by this fix.

Try [the five public engine projects](<../examples/opencv tools/dnn_engines/README.md>).
They include the same RGB-mean bytes under Classic/New/Auto, a New proposal pipeline
and an intentionally rejected Classic proposal pipeline. The fixtures are original
MIT code/images rather than trained recognition benchmarks. Native discriminator,
frozen-byte/race, cleanup, cache/reset, concurrent selection, exact regeneration,
JSON/XML/Save As and cold-worker checks qualify this CPU reliability increment.
It adds no completed vision capability to the roadmap counts.
