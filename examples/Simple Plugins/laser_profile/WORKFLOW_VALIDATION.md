# Dev2 workflow validation — 2026-11-16

- **67 focused tests pass** in the host environment. Coverage includes source/
  metric parity, encoding/grid rejection, unavailable results, recipe cache
  invalidation, Save As asset copying, private teaching snapshots, novel
  reference training, native zoom/reopen, independent previews, stale/closed
  job cancellation and Stage/Cancel behavior. Ruff formatting/checks pass.
- The **actual host inspector** was exercised offscreen: restore original
  training → edit region → learn → Stage → verify applied configuration unchanged
  → Apply → reopen. The reopened source ROI `[280,20,80,400]` produced 400 valid
  metric bins. The original recipe stayed byte-identical. No desktop input or
  screenshots were used.
- Six saved execution projects validate, save, reopen and run. The same 90%
  availability gate is used for every case: generated step passes with 480 bins;
  generated missing/competing/input-ceiling case fails with 390; the four real
  WLI-Set cases fail with 72/42/2/7 out of 120. Reports match standalone results
  exactly apart from the explicit decision field. No per-image relearning.
- Explicitly loading the published plugin uses its bundled **dev2 wheel**
  without installing anything into the host environment. All six projects were
  replayed from that wheel with graphics on and off. Counts, reports, decisions
  and measured clouds remain identical; colors/rasters/drawings are skipped off.
  The cache is reused. Overlay/chart outputs are retained as actual image results.
  The packaged popup restores its reference and Cancel preserves configuration.
- Dev2 wheel SHA-256:
  `97433ebfe03f775fb8b7c43173b18682b1cf7bd2b16333d1c3d965b209719f06`.
  Eleven runtime modules; MIT license and CLI entry point are retained. E1
  studies and the original dev1 wheel remain available separately.

Calibrated clouds contain only accepted measured points with camera frame,
meters and optional RGB. Source-pixel links round to the nearest pixel; full
subpixel centers remain in the report. Pixel-only and empty results emit no cloud.
The earlier native fly-around rendering check remains recorded in
[E1 validation](VALIDATION.md).

These are software/packaging/workflow checks. The generated 50-mm step remains
a software control, and WLI centerlines remain industrial-software references.
Neither establishes optical calibration, physical accuracy or multi-sheet
identity. Future qualification and enhancements are listed in the tool roadmap.
