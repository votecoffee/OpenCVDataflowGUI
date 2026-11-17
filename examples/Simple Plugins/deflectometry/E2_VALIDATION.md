# SimpleDeflectometry E2 validation — 2026-11-17

Scope: retained wrapped-phase reference inspection. Known-plane reflection math
remains a separate API. Absolute screen decoding, physical calibration/accuracy
and measured shape are open follow-ups.

**50 focused checks pass** against both installed and checkout host APIs.
The combined checkout run passes **105 checks**, including all 55 existing
FringeScan checks. The isolated OpenCV4.12 numerical environment checks the 32
non-host cases; floating comparisons to a different NumPy/BLAS environment allow
1e-12 roundoff, while masks/reasons/counts are exact. Packaged host replay is
exact for every raw map and the full report. Ruff and TOML 1.0 parsing pass.

Retention checks cover private arrays/nested JSON, exclusive save, archive/member
hashes, duplicate/object/oversized declarations, exact masks and unknown-exposure
code diagnostics. Runtime checks cover source/grid/order/response/exposure,
inspection failure versus invalid input, no hidden source fallback, cache file
changes, reference/basis/sequence reuse, graphics-off parity and Save As bytes.

Native checks cover a restored 12-frame filmstrip and selected frame, exact ROI/
mask and phase settings, zoom/draw/edit modes, image-file teaching without a
manifest, shift/order controls, separate reference health, independent failed
previews, stale/closed workers, private connected snapshots and stale publication.
The real 8-frame SOFAST reference restores its exact mask/null exposure and retains
the upstream license/provenance.

The actual host inspector runs offscreen with the packaged wheel. Stage changes
only its draft; Apply commits the new asset; reopening restores frame 10 and the
40x30 ROI with 1,200 comparable pixels; a subsequent Cancel preserves applied
training. No desktop input or screenshot is used. See
[e2_inspector_checks.json](e2_inspector_checks.json).

The dev2 wheel has 10 runtime modules and uses the same pinned 13-module FringeScan
wheel. Every module matches source bytes. Four independent CLI replays match
all 18 raw maps and full reports exactly. See
[e2_package_verification.json](e2_package_verification.json).

Four one-block execution graphs use the same generic fit defaults, 95% comparable
coverage and illustrative 0.45-radian phase-vector limit. All three analytic cases
share the clean retained reference. Clean 6,912 PASS; localized change 6,912 expected
FAIL; missing 6,212 expected FAIL; SOFAST 114,356/114,359 PASS code self-diagnostics.
No held-out real defect or metric-height truth is claimed. Reopen/run, 16 named
RGB views per result, raw/report parity, graphics-off and cached-reference reuse
are recorded in [e2_published_checks.json](e2_published_checks.json).

Explicit plugin loaders accept an already loaded **identical checksum-pinned**
shared wheel from another example folder. Both FringeScan→Deflectometry and
Deflectometry→FringeScan load orders register and validate without processing;
different wheel contents remain rejected. See
[plugin_coexistence_checks.json](plugin_coexistence_checks.json). No shared runtime
API or installed dependency is changed.

All 19 published original image/README members and original data-license bytes
remain unchanged. New real-data sealed training assets retain CC BY-SA 4.0 and
upstream attribution; numerical code remains MIT. Old E1 reviews/receipts and
the dev1 wheel are preserved. The first native phase-reference workflow gate is
met; [remaining work](FUTURE_WORK.md) stays explicit.
