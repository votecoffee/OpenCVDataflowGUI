# SimpleFringeScan E2 validation — 2026-11-16

Scope: retained setup and cached execution with imported known calibration or
explicit nonmetric phase/correspondence. Software/GUI checks do not establish
acquired-camera accuracy or physical synchronization.

## Focused checks

**55 checks pass** in the actual host Python 3.14.3 environment (OpenCV 5.0
contrib, NumPy 2.5.4, PySide6). This includes the 25 original numerical checks
and 30 retention/execution/native-editor checks. No host dependencies changed.

- Independent phase/triangulation controls, lens distortion, native Gray ordering
  and fallback parity, parallel/invalid ray rejection and graphics-off parity.
- Private immutable retained JSON/arrays/mask, sealed recipe hashes, exclusive
  save, malformed/object archive rejection and captured source ownership.
- Connected order/grid/exposure/sequence/calibration checks, recipe file-change
  detection, compiled phase/camera-ray reuse and exact raw-map parity.
- Native source/ROI/calibration restoration, zoom, independent preview, stale-job
  suppression, close/cancel, phase-only behavior and connected private snapshots.
- Host Save As retains sealed asset bytes. Opening/validation does not load a
  recipe or execute processing. Failure results retain valid evidence; invalid
  input does not leave a stale cloud.

## Actual inspector and packaged execution

The actual host inspector ran offscreen using the packaged wheel. Stage changed
only the draft; Apply committed the new recipe; reopening restored the edited
40x30 ROI and 1,200 accepted pixels. A subsequent Cancel preserved the applied
recipe. No desktop input, screenshot or hardware activation was used.
See [inspector receipt](e2_inspector_checks.json).

The dev2 wheel has 13 runtime modules matching source bytes. Its three CLI
replays match every retained E1 map exactly, including NaN and discrete reasons:
18 raw maps for each metric case and six for phase-only. See the
[package receipt](e2_package_verification.json).

Three one-block execution projects reopen and actually run the packaged plugin.
The clean/missing controls share the clean retained reference. Identical generic
settings and a 95% availability gate give these expected outcomes:

| Case | Accepted | Coverage | Expected judgment |
| --- | ---: | ---: | --- |
| Analytic clean | 6,912 | 100% | PASS |
| Analytic missing | 6,415 | 92.81% | FAIL, evidence retained |
| External simulated crown, phase-only | 56,985 | 10.18% | FAIL, no metric cloud |

Metric results carry native fly-around clouds in camera meters, source pixels
and ray-gap colors. They have eight RGB review images; phase-only has six and
emits no XYZ. Graphics-off retains identical maps, report, points and source
pixels, skips colors/rasters and reuses the same compiled model. The
[published execution receipt](e2_published_checks.json) records replay.
All 15 original attributed dataset members remain hash-identical; PBRT scene
files are preserved as binary assets to prevent newline normalization.

The source [workflow guide](README.md) and
[remaining work](FUTURE_WORK.md) distinguish imported calibration from future
calibration estimation. Real acquired data, independent gauge/shape accuracy,
radiometry/motion qualification and tiled/live execution remain open. The
known-calibration software workflow gate is met; those milestones stay open.
