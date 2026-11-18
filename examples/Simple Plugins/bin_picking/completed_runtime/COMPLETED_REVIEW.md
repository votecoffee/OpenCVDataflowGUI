# Completed bin-picking evidence

Run a suction or CAD-jaw block in **Debug**, then use its inspector's
**Review completed bin-picking evidence…** button. This opens an independent
native fly-around viewer, separate from training. Review uses the paired finished
inputs, effective bin transform, executed recipe and recorded candidate decisions.
It never reruns planning, depth projection, contact tests or collision checks.

The table includes every recorded candidate: eligible, rejected and unknown.
Select a row to inspect its geometric score, contact fraction, clearance state,
reasons, collision witnesses and unknown query locations. **Focus candidate** hides
the scene/bin context to fit the selected geometry; **Overview** restores context.
Layer controls change display only. The overview draws a bounded subset of
alternatives; it does not discard table rows or change ranking.

The native viewer supports yaw/pitch, point size and fly-around. Green is eligible,
red rejected, orange unknown, magenta observed collisions; TCP axes use RGB.
Suction highlights a selected cup in yellow. CAD models, cups, jaws, sweep envelopes
and pose axes are derived configured geometry, distinct from measured first returns.
Unknown query locations do not establish free space. Grasp scores describe geometric
evidence, not pick success probability. Eligibility does not qualify a robot move,
force closure, hidden contacts or the transport of a held part.

**Review captured images…** opens read-only depth/instance/appearance inputs with
wheel zoom and drag pan. Depth contrast is normalized for display only. The original
array dtype, invalid values, units, display attributes and metadata remain retained.
An appearance input is not assumed registered to depth. Declared capture IDs and
software completion tickets do not prove physical sensor synchronization.

Jaw pose-fit and mesh-fit views use the exact diagnostic arrays retained during
execution. A run with graphics disabled does not perform extra diagnostic work for
review. Its plan view can be drawn later from the already recorded decisions and
measured points, but missing pose/mesh diagnostic arrays remain unavailable.

**Export frozen review…** writes a new folder containing the original byte-identical
`.sbp`/`.sbpj` archive, complete candidate report, exact native input and scene NPY
arrays, emitted cloud arrays/PLYs, retained diagnostic arrays, current selected PLY
and a labeled assessment PNG. PLY/display use the host's floating-point representation;
the NPY files preserve original native arrays. Export never overwrites a folder.
The snapshot can be exported after the graph closes or the original recipe disappears.

A new editor publication marks an open snapshot stale without changing it. Use
**Capture latest completed run** explicitly. Active, errored, unexecuted or Lean runs
cannot supply a new review. A failed recapture leaves an existing frozen snapshot
visible with an explanation. Training drafts and Apply/Cancel remain independent.

Execution retains a small binding, serialized numeric report, references to completed
evidence and the model bytes already read on a cold recipe load. Full array freezing,
image display and export happen only on explicit review. Lean does not retain this
review record. The review adds no projection, collision, diagnostic or robot work to
the execution path.

See [public completed-run examples](README.md) for real
T-LESS replays and generated first-surface controls. Real examples retain their
original conservative rejection gates; they are not physical grasp ground truth.
