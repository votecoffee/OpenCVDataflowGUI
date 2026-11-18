# Capture completed preview inputs

`0.1.0.dev8` adds **Capture last processed inputs** to the Simple3DLocate
trainer. It takes the search cloud, provided camera and provided full-frame
depth cloud from one completed execution published by the host. It does not
combine separate reads of the latest upstream outputs. This is useful when
reviewing a failed or low-scoring inspection after acquisition has advanced.

## Workflow

1. Save the host project, wire `scene` and any `camera` / `depth_scene` inputs,
   and run in a profile that retains inputs, such as **Debug**.
2. Open **Train 3D locator…** and choose **Capture last processed inputs**.
3. Review **Test search cloud** and **Verification depth cloud**. The latter
   uses the search cloud when no separate depth input was provided.
4. Choose **Locate in test scene**, or compare execution profiles. Capturing
   inputs preserves the reference, trained samples, settings and compiled model.

The retained run can have a failed inspection judgment. An exception without a
completed result has no completed inputs to capture. The preview source label
shows the completed run count, its judgment, provided inputs and search/depth
ticket status. Its tooltip gives snapshot and per-port ticket details. The
private snapshot retains the original per-port payloads for inspection.

The host needs `snapshot_processed_inputs`. On older hosts this button is
disabled; file loading and the existing manual captures remain available.
Switching to Debug alone cannot recreate earlier discarded history: run again.
Capture does not trigger a new graph run or change the inspector draft.

## Validation and replacement

All captured values are validated before replacing any preview input. Both
clouds need finite bounded XYZ, valid provided normals, declared units and
frames. Camera and separate depth units/frame must match the search cloud.
There is no implicit unit conversion or coordinate transformation.

Absent optional camera/depth inputs **clear** their previous preview values.
This prevents a new search scene from silently inheriting another run's depth
or calibration. A validation failure, unavailable history or busy-block error
leaves the earlier preview inputs, result, provenance and trained model intact.
It never falls back to reading newer upstream values independently.

The host returns private copies. Changing upstream arrays, payloads or outputs
does not change an existing preview. A later button press captures the latest
completed locator execution; a new upstream output alone is insufficient.
No extra cloud copy is made by the trainer after host capture. Validation and
display work happen only on capture; matching and its runtime inputs are unchanged.

## Execution identity and acquisition identity

One completed execution can consume inputs from different acquisitions,
especially with latest-input coordination. Snapshot IDs identify private copies,
not sensor exposures. Execution counts can reset and output revisions can
advance because another block publishes. None proves acquisition pairing.

For search and separate measured depth, the trainer compares the host's typed
ticket keys `(session, kind, value)`. Timestamp tickets are normalized to UTC;
different source names or sequence counters do not override the host's identity
semantics. The display distinguishes matching, different, missing/unrecognized
and explicitly mixed-ticket coordination metadata. Matching tickets report
agreement, while physical capture timing remains unverified. A camera
calibration ticket is retained separately; calibration may be static and need
not share the measured clouds' ticket.

Different/missing tickets do not block diagnostic replay of what the locator
actually consumed. Matched saved files and registered static scenes can have
different automatically issued tickets. For moving scenes, arrange common
acquisition identity and appropriate input coordination upstream; a trainer
snapshot does not synchronize cameras or repair input pairing.

Individual **Capture latest…** buttons and file loads remain independent
operations. Changing or clearing any preview component clears the completed-
capture provenance label. It does not keep claiming that edited inputs came
from the original bundle. These preview inputs and their snapshot metadata are
ephemeral; Stage stores the trained recipe, not acquisition history. Runtime
still needs the actual scene/camera/depth wires. All existing recipes retain
their matching behavior and file format.
