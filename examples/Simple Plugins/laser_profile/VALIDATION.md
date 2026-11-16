# E1 validation — 2026-11-16

- **44 focused tests pass**, covering geometry, held-out 50-mm image reconstruction,
  multiple/overlapping peaks, missing/clipped/input-ceiling states, original image
  coordinates, frozen learning, source ownership, calibration/recipe round trips,
  bounded malformed arrays/archives and train → inspect → execute CLI replay.
- Ruff checks and formatting pass. TOML 1.0 parses with Python `tomllib`; the
  package manifest retains UTF-8, no BOM, LF and one final newline.
- Dev1 wheel SHA-256:
  `70e9d3ee0e04f989b56c19feb7715973c31e1eefb40249ad5c5f971ef83eb75b`.
  All seven packaged runtime modules match source bytes; MIT license and console
  entry point are included. The wheel was loaded separately from source without
  changing the host environment. All seven reference/test numerical reports
  match the retained source reports exactly with graphics disabled.
- All **eight** saved host review projects validate, save, reopen and load their
  original/result sources. Calibrated clouds have **480, 480 and 390** measured
  XYZ points, meters, camera frame and RGB. Real uncalibrated reviews have no
  cloud. Native Qt cloud rendering and orbit change were checked offscreen;
  no desktop input, screen capture or physical device was used.
- WLI-Set version 3 license metadata was fetched from its versioned Figshare API
  and verified as **CC BY 4.0**. Selected source/member hashes are retained.
  The two poor fixed-ROI fits remain visible and do not count as successful
  stripe localization. Pixel-centerline agreement does not qualify physical Z.

Numerical tests used Python 3.13 with OpenCV 5.0.0.93; wheel/native review used
the host's Python 3.14/OpenCV 5.0 environment. The OpenCV 4.x iterative branch is
implemented, without a separate platform qualification in this increment.

Still open: guided teaching popup/execution plugin, image-based calibration,
independent physical gauges, multi-sheet identity, scan accumulation and broader
performance qualification. This milestone does not close the first delivery gate.
