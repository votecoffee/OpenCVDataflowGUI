# Suggest and review opposing CAD contacts

SimpleBinPicking Jaw dev11 adds **Suggest CAD contacts…** to the training editor.
It proposes part-relative grasps from retained surface points and outward normals.
It never inserts a proposal automatically or accepts a runtime grasp.

## Training workflow

1. Restore/import the part and enter the real gripper stroke, finger, pad and TCP
   dimensions. Teach the desired **+Z retract** preference on the current grasp.
2. Choose **Suggest CAD contacts…**. A bounded background worker finds opposing
   normal patches, ranks finite pad support and checks sampled CAD sweeps.
3. Select rows to inspect the native colored fly-around cloud. Gray is retained
   CAD, yellow marks the actual two source-row contacts, green/red crosses show
   supported/missing finite pad tests, magenta marks CAD collisions, blue shows
   sweeps and RGB axes mark the TCP. Reviewable means model checks passed.
4. Explicitly **Add selected to training draft**. Rejected rows cannot be added.
   Existing grasps, retained mesh, original frame and frozen references survive.
   Added contacts retain source-row provenance in sealed recipe notes.
5. Preview current/reference depth and poses, then **Stage recipe → inspector
   Apply**. Adding invalidates the earlier preview. Undo removes the addition;
   cancelling either popup or the training inspector preserves applied training.

Multiple rows can be selected, up to the recipe's existing 32-grasp limit. The
popup shows the first selected cloud. Review/orbit uses cached results, without
searching again. Cancel signals the worker and discards incomplete proposals.

The current grasp's retract vector is a declared preference in the **original
part frame**, not a camera/world-axis guess. It is projected perpendicular to
each actual closing line. Change the taught retract before searching if another
approach is required. The runtime still checks approach tilt in the current bin.

## Search and evidence

The generic search uses dominant antipodal normal bins, with at most 12 closing
axes. A fixed 5×5 tangential target grid samples opposing patches per axis. At
most 300 source contact pairs are seeded, ranked using the same finite 3×3 pad
tests as the runtime, and at most 96 receive full sampled-model sweep checks.
Up to 24 spatial/orientation-distinct suggestions are shown by default.

Seeds outside the mechanical stroke are retained as rejected evidence; reachable
seeds are prioritized for the bounded full checks. Contact angle, distance, pad
coverage, opening margins and model collisions use the retained gripper settings.
Search budgets, normal bin width/separation, source geometry hash, frame, retract
and gripper settings are exported by the API report. There are no per-example
thresholds or dataset-specific branches.

This is an approximate candidate search. Normal bins, the fixed anchor grid,
surface sampling and candidate budgets can miss usable contacts. It is not
exhaustive, a grasp optimizer, a force-closure test or a physical success model.
No opposing normals means no invented back surface. Small/curved patches and
oversized parts remain rejected when current finite-pad/stroke checks fail.

Model-reviewable contacts still need accepted current pose evidence and measured
scene clearance. Hidden CAD geometry is model assistance, never an observation
or proof of free space. Measured occlusion/missing space stays unknown; neither
the proposal search nor the mesh diagnostic bypasses the existing runtime gate.
Robot IK, continuous collision freedom and transported-part volume remain
separate qualification work.

## API and compatibility

```python
from opencv_simple_bin_picking import propose_contacts, add_proposals, proposal_cloud

review = propose_contacts(recipe, retract=(0, 0, 1), cancelled=cancel_event.is_set)
report = review.report()  # JSON-compatible, acceptance_applied=False
selected = review.proposals[0]  # choose explicitly after inspection
graphic = proposal_cloud(recipe, selected)  # derived RGB cloud in original part frame
draft = add_proposals(recipe, [selected])  # rejects failed/stale/conflicting selections
```

`add_proposals` rechecks source contact coordinates, current pad/stroke/model
checks and unique names; the original recipe is unchanged if validation fails.
Dev12 also checks the exact source-contact span, rejecting a changed span even
when its unchanged frame and current contact tolerance would otherwise pass.
The UI additionally binds the result to the unchanged training draft. A changed
source, settings or name requires rechecking; adding never runs a robot.

Point-only/version-1 and mesh/version-2 archive formats are unchanged. Suggestions
use ordinary retained grasp records and bounded JSON notes. Runtime execution
does not generate proposals, and existing trained recipes incur no search work.
No new mandatory dependency or host SDK/environment change is required.
