# SimpleFringeScan remaining work

The first retained known-calibration setup/execution workflow is delivered.
The following work remains separate from that software gate.

- Qualify licensed acquired camera/projector sequences against independent
  shape or gauge measurements. Present rendered controls as numerical evidence.
- Estimate camera/projector calibration from image observations with residuals,
  uncertainty, held-out checks, frame/unit review and retained source images.
  Current calibration is imported, reviewed and editable.
- Calibrate projector/camera response and qualify exposure variation, motion,
  defocus, specular leakage, ambient changes and depth discontinuities.
- Add stronger coding-error/phase-unwrapping checks and multi-frequency sequences;
  a low sinusoidal residual does not prove absolute correspondence correctness.
- Add bounded tiled/streaming execution, shared reusable calibration caches and
  measured multicore throughput. Current vectorized execution caches phase-fit
  basis and camera rays and supports parallel independent block instances;
  no measured cycle-time gain is claimed for those caches yet.
- Add richer source-mask editing and acquisition adapters. The current editor
  retains an imported exact mask and supports a zoomable rectangle; drawing a
  replacement rectangle intentionally replaces that mask.
- Add project-contained retention for external runtime manifests/images during
  export. Save As carries `.sfs` training assets, while ordinary manifest paths
  remain references to their image files. Keep the capture folder intact.
- Add failed/accepted point selection and source-linked cloud probing if the
  host later supplies that native capability; never fabricate points for gaps.

No gap filling, wrapped-phase-to-height shortcut or acquired-camera accuracy
claim is part of the current release. Advancing to the next roadmap tool does
not close these qualification and scalability milestones.
