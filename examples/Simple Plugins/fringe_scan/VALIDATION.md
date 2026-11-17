# Initial FringeScan validation — 2026-11-16

**25 focused checks pass** in isolated OpenCV 4.12 contrib and the actual
OpenCVDataflowGUI Python 3.14 / OpenCV 5.0 contrib environment. They include
native Gray generation/ordering and every projector pixel in a non-power-of-two
control, independent N-step phase/amplitude construction, missing/clipped/dark/
weak observations, capture order/radiometry/ROI rejection, cancellation, metric
XYZ against independent projected points and OpenCV triangulation, distortion,
parallel/outside/mismatched rays, rigid-frame/unit/depth checks, bounded imports,
phase-only scope and graphics-off numerical parity.

Two analytic ray examples retain 6,912 and 6,415 accepted points with explicitly
known camera/projector calibration, no gap filling and one visible surface per
camera ray. Their maximum independently constructed rendering-reference error
is about 1.04e-8 m; this is a software control, not physical accuracy evidence.
The missing case retains 497 rejected camera pixels and their raw reasons.

The external CC BY4 Warsaw crown example has 56,985/559,872 phase pixels available
under unchanged generic settings. Low modulation and poor fits remain visible.
Its original images are **simulated**, its sRGB preprocessing is a documented
standard-PBRT inference, and it has **no imported metric reconstruction or shape
ground truth**. It emits no metric XYZ/cloud. See dataset credits and transforms.

The dev1 wheel contains seven runtime modules matching source bytes, MIT license
and a console entry point. Three independent wheel CLI replays match the retained
numerical maps; exact equality/tolerance and hashes are recorded in
`package_verification.json`. No wheel installation into the host is needed.

Three native saved review projects validate, save, reopen and load their retained
views. Both metric reviews preserve accepted count, colors, meter units, camera
frame and XYZ in the native fly-around point-cloud value. The phase-only review
has no cloud node. These projects are **file-only loading reviews**; estimator
reexecution is proved separately by wheel CLI replay. The host review receipts
and published replay receipts retain this distinction.

Ruff and TOML1.0 checks pass; text remains UTF-8/LF and original binary inputs
stay exact. No host dependency changes, live acquisition, desktop interaction or
automatic CI trigger was added. The retained setup/editor and execution plugin
are still next; acquired-camera accuracy remains open.
