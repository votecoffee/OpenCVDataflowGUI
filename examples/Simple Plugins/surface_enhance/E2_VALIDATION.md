# Retained setup and execution validation — 2026-11-16

**54 focused checks pass** in the actual host Python environment, including
recipe round-trip/source ownership, bounded archive/import rejection, declared
exposure changes, calibration/order/grid rejection, compiled-cache reuse,
graphics-off numerical parity, unavailable inspection results, exact Save As
assets and actual connected array-bundle snapshots. Native Qt checks run
offscreen and exercise restore, all light images/calibration, zoom, exact masks,
region edits, independent previews, stale work, cancellation and worker closure.

An additional actual host inspector check confirms **Stage changes only the
draft**, Apply commits the new recipe, reopening restores the edited 100×90
source ROI and its 9,000 available pixels, and Cancel preserves the applied
recipe. No desktop input, screen capture or hardware acquisition is used.

Five saved execution projects reopen/run with the same generic fit settings
and one 95% availability threshold. Four pass; the ambiguous four-light case
fails availability while retaining its 11,952 valid pixels and all raw/image
evidence. Every case exactly matches all 14 retained standalone numerical maps.
Graphics-off runs retain identical maps/reports and reuse the compiled estimator.
The five original file-only image-review projects remain separate loading checks.

The `0.1.0.dev3` wheel's ten runtime modules match source bytes and include the
MIT license/console entry point. Independent wheel CLI replays exactly reproduce
all retained maps. `e2_package_verification.json` records the wheel/module hashes;
`e2_published_checks.json` records actual bundled-plugin project replay and
`e2_inspector_checks.json` records the native inspector workflow.

The numerical core loads no Qt/host runtime; the setup factory imports Qt only
when explicitly opened. The trusted bundled plugin checks the wheel hash and
requires no host environment installation. No automated CI trigger was added.
Ruff and TOML1.0 checks pass; text remains UTF-8/LF and binary inputs stay exact.

This completes a first supported workflow for registered linear images and
known directional lights. It does **not** qualify real camera/material accuracy,
physical height, measured registration, automatic light calibration or defect
acceptance. The examples are analytic and attributed CC0 material-map **rendered
controls**. Acquired real multi-light validation and richer models remain open
follow-ups as the shared roadmap advances to FringeScan.
