# Restore a saved background reference

`sequence.mog2` and `sequence.knn_background` can reconstruct their initial
background from a sealed clean image before processing the first current capture.
This is bounded reference replay, not serialization of an opaque native model.
Native mixture history, KNN random scheduling and a multimodal scene history are
not saved. Try the [eight examples](<../examples/opencv tools/background_reference/README.md>).

## Capture and use a reference

1. Connect a clean image to **Capture clean background reference**
   (`sequence.capture_background_reference`). Set a stable **Camera ID** and an
   optional **Calibration ID**. A connected **Known foreground** mask containing
   any marked pixel rejects capture. Unmarked objects cannot be ruled out by the
   recorder; use a genuinely clean scene.
2. Connect its **Reference** to **Save background reference**
   (`output.background_reference`). Choose the destination JSON. **New** preserves
   an existing file; **Replace** publishes through an adjacent temporary file and
   atomic replacement. File output requires an explicit run and has no live preview.
3. On MOG2 or KNN, set **Reference path**, **Reference camera ID** and
   **Reference calibration ID**. IDs must match the record exactly. Optionally pin
   the file's exact SHA256, reported by the save block.
4. Run the inspection. A new native model replays the clean reference, verifies
   background classification and exact background-image recovery, then evaluates
   the current capture. A verified seed bypasses ordinary live-image warmup.

These tools are under **OpenCV > Video Analysis (video)**. The file field is a
sealed project asset: Save As copies it without changing pinned bytes. The
creator, saver and native consumers are also available through the ordinary
headless graph API. Public Python helpers are in `background_reference.py`.

## Binding and persistence

The versioned `background_reference_v1` record contains a lossless
`trained_image_v1` image, declared camera/calibration IDs, recipe policy and nested
checksums. Image data uses bounded NPY without pickle. Supported references are
gray, RGB or BGR uint8, or float32 in native [0,255] intensity units. KNN uses
uint8; MOG2 also accepts float32. References retain original fractional samples.
Image data is limited to 8 MiB and formatted JSON to 12 MiB. Loading rejects
duplicate fields, NaN/Infinity, BOMs, unsupported schemas and corrupt pixels or
checksums. Input pixels must be finite.

Current size, depth, channel order, spatial grid and pyramid geometry must match.
Acquisition root labels may change after restart; root extents and transforms
must match. **Strict** recipe binding also checks recipe context. **Ignore** is an
explicit capture-time option for backgrounds shared across recipes. Changing
the camera/calibration ID, grid, encoding or bound recipe rejects the reference
before native model allocation. These are declared setup identities and numerical
geometry checks, not authentication of a physical camera.

The runtime caches a validated file snapshot and decoded image, not the large
JSON/base64 record. File identity/size/timestamps and the expected hash select the
cache. Changed file metadata triggers reloading; **Reset** always rereads it.
Edits that preserve all file metadata require Reset to be detected. Embedded
checksums verify every loaded snapshot, and the optional SHA256 pins its exact
bytes. Missing or invalid files do not fall back to a previously loaded model.
Reference changes participate in the sequence context and reset the model.

## Frozen and guarded learning

**Reference update = Frozen** is the default. Current images do not become
persistent background observations. MOG2 can retain its frozen native model.

The supported wheel's CPU KNN implementation can update samples even at a zero
learning rate. Frozen KNN therefore builds a disposable model from the clean
reference for each current classification, snapshots its verified background,
uses a positive native query rate and discards the queried model. The logical
background stays frozen. This costs reference replay on every activation; choose
MOG2 when its model fits the application and avoiding that cost matters. The
status field `frozen_query_rebuild` makes the behavior visible. No global RNG,
OpenCL or thread configuration is changed. The upstream
[KNN implementation](https://github.com/opencv/opencv/blob/5.x/modules/video/src/bgfg_KNN.cpp)
shows the native update path; local repeated-part tests qualify the workaround
for this supported wheel.

**Guarded** permits adaptive learning but requires a current-grid uint8 **Known
foreground** mask on every activation. Marked pixels are replaced by trusted
background values before native learning and remain foreground in the output.
This also protects against the wheel's KNN mask option updating samples before
marking those pixels. Supply correct masks from the first activation; the tool
cannot undo contamination from objects that were previously left unmarked.
This protection applies to the restored-reference workflow. Unseeded native
models retain their existing behavior and do not acquire this qualification.

The ordinary **Mask/region** inputs select reported foreground; they do not
restrict the native model's training domain. Use **Known foreground** for learning
protection. Foreground pixels retain the original current encoding and values;
native background images use grayscale/BGR. Outputs keep the current capture's
ticket, geometry and acquisition payload, rather than saved-reference identities.

## Readiness, limits and diagnostics

Replay uses configurable passes and learning rate. Before allocating a native
model, `pixels * (passes + 1)` must fit **Maximum reference work**; the extra pass
verifies reconstruction. Frozen KNN pays this work per activation. Native model
budget reservations include a conservative reference-image allowance; one cached
reference is separately hard-limited to 8 MiB of pixels. File parsing and native
working images require temporary memory as well.

Replay must classify the entire clean reference as background and reproduce its
background samples exactly. If it cannot, increase passes or review the
reference/model settings. A failed probe, missing guarded mask, bad configuration
or mismatched record emits no primary data. Native errors are contained by the
common executor and failed sequence state is retired. A later valid run can
recover. A connected Known foreground input that emits no value holds the block
rather than silently running without that mask.

Sequence status includes reference ID, file hash, replay count, verification,
update policy and query-rebuild behavior. Activations and tickets count current
images, not internal replay passes. Reset/reopen restores from disk; native models
and file caches are never serialized. Graphics/profile, timing, logger and event
behavior use the common runtime.

Qualification includes independent rectangular foreground truth, native clean-
replay comparisons, fresh acquisition roots/current tickets, persistent-part
tests, camera/calibration/grid/recipe/format rejection, file corruption and pins,
cached refresh/reset, guarded requirements, insufficient replay and work budgets,
atomic publication, XML/Save As/reopen and offscreen controls. The original
fixtures qualify software behavior; they do not establish that a real reference
scene is clean or that an inspection is production-ready.
