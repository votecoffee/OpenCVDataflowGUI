# SimpleDeflectometry E1 validation — 2026-11-16

Scope: offline registered temporal phase reference comparison and separate
known-plane reflection math. Native teaching/execution integration is next.
Absolute screen decoding, absolute shape and physical accuracy remain open.

## Numerical and input checks

**23 focused checks pass** in host Python 3.14.3 / OpenCV 5 contrib / NumPy 2.5.4
and isolated Python 3.13 / OpenCV 4.12 contrib / NumPy 2.2.6. The combined host
run passes all **78 checks**, including the existing 55 FringeScan checks. Ruff
passes. No host dependency was installed or changed.

- Independent circular differences across phase seams, sign conventions and
  explicit invisibility of whole-period changes. Reference/test invalid sets,
  clipping/nonfinite/weak/poor-fit reasons and retained ROI denominators.
- Private readonly capture arrays/masks, source order/grid/setup/exposure
  mismatch, uncalibrated-code scope, explicit radiometry and three-step lack
  of redundant fit evidence. Duplicate/nonfinite JSON and object NPY rejection.
- Cached reference fits and phase bases reused across captures; graphics-off
  exact map/report parity and preserved existing output directories.
- Independent reflection-law normal controls at 0, +5, -10 and +15 degrees,
  central-ray half-angle, slope sign, frame/screen/plane validation, off-screen,
  unavailable/coincident/parallel/behind/grazing/non-graph cases. Assumed plane
  points are retained as assumptions, not reconstructed measured shape.

## Retained studies and software truth

| Case | Comparable pixels | Estimator outcome | Ground-truth scope |
| --- | ---: | --- | --- |
| Analytic clean | 6,912/6,912 | PASS | Independently generated unchanged phase. |
| Analytic change | 6,912/6,912 | FAIL phase-vector limit | Known localized circular phase change. |
| Analytic missing | 6,212/6,912 | FAIL availability | 700 unavailable pixels remain gaps. |
| SOFAST self-review | 114,356/114,359 | Code-diagnostic self-comparison | Actual exported camera frames; no held-out change or metric truth. |

Generic fit settings, 95% comparable-coverage gate and illustrative 0.45-radian
phase-vector limit are identical across studies. Analytic raw change agrees with
independent truth within 2e-7 rad after float32 import. The real self-comparison
has zero change by construction and is labeled uncalibrated codes/null exposure.
Its value is import, fit, unavailable-state and replay evidence.

All 18 retained original PNGs plus README match pinned ZIP member hashes. The
83,154,541-byte ZIP matches publisher SHA1 and retained SHA256. Separate original
CC BY-SA 4.0 license bytes match publisher file SHA1. A full independent repeat
of the source preparation matches published source files. Upstream processed
HDF5 is inspected numerically only; its computed shape is not physical truth.

The two packaged wheels select the exact tested source modules. Four independent
CLI replays from those wheels reproduce all 18 raw maps per case and the complete
reports exactly, with graphics disabled and no metric export. See
[package_verification.json](package_verification.json).

Four native saved projects validate/save/reopen/run using existing image sources.
All **64 result image loads match PNG bytes numerically**. File-source PASS
means loading only; decoding is not reexecuted by those graphs. See
[host_review_checks.json](studies/host_review_checks.json). These reviews use no
plugin, equipment, hooks, desktop input or screenshot. The standalone launcher
explicitly selects two trusted bundled wheels without installation.

## Remaining work

Retained native teaching with editable region, independent previews, Stage/Apply/
Cancel and cached host execution are next. Then verify coded/multi-frequency
absolute screen correspondence and obtain registered real reference/test captures
with original exposure/response and camera/screen calibration. Qualify reflection
geometry, repeatability and physical surface/change truth independently. The
first intuitive supported-workflow gate stays open until native teaching and
execution are delivered. No measured cycle-time improvement is claimed yet.
