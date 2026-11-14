# Vision application plan

Use the sections relevant to the application. Replace unknowns with explicit
assumptions and experiments; do not invent calibration, image rights or accuracy.

## Inspection objective

- Part/process and operating conditions:
- Required outputs (detections, poses, measurements, images, final judgment):
- Acceptance limits, units, failure reasons and ambiguous/unknown behavior:
- Latency/throughput target and what its measurement includes:

## Inputs and evidence

- Camera/files/video/depth/cloud inputs, dimensions, encoding, bit depth and rate:
- Trigger/ticket identity; external encoder, robot pose or recipe association:
- Representative image set, labels, licensing/private status and expected outcomes:
- Calibration: intrinsics, distortion, pixel-to-world mapping, frame names and units:
- Training/reference assets, immutable versions and reproducible load requirements:

## Graph proposal

| Step | Existing block ID or ToolBlock | Inputs → outputs | Important settings/constraints | Evidence needed |
| --- | --- | --- | --- | --- |
| Acquisition | | | | |
| Preprocessing / region selection | | | | |
| Locate / align | | | | |
| Measure / inspect | | | | |
| Aggregate / final judgment | | | | |
| Report / integration | | | | |

Record conversions, spatial/pyramid provenance, pose-relative regions, input failure
policies and fan-in/ticket matching. Specify any buffering, bounded parallelism,
warmup, timeouts and ordering. Explain which outputs require result graphics.

## Capability gaps and implementation

Distinguish existing baseline, missing workflow configuration and truly missing
tools. Prefer a reusable ToolBlock for a repeated graph. Use a Python plugin for
specialized algorithms; give it a stable type ID, ports, controls, validation,
failure behavior, training persistence and focused checks. Record optional packages.

## Acceptance and handoff

- Working project path and source revision; independent files for concurrent edits:
- Offline expected pass/fail/unknown cases and preferred failure reasons:
- Numeric truth/tolerances and geometric/visual review points:
- Focused software tests and headless commands/reports/previews:
- Performance measurements after correctness, with diagnostic settings recorded:
- Any camera/robot/PLC or physical checks still needed, separate from offline proof:
- Deliverable project, assets/plugin setup, concise usage and application limits:
