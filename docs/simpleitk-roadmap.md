# SimpleITK medical imaging roadmap

SimpleITK is an optional, separate image/volume backend. Its work is tracked
independently from OpenCV's four coverage stages and from scikit-image. The
checklist below describes selected application workflows, not exhaustive coverage
of the SimpleITK API. [Current contracts and setup](simpleitk.md).

## M1 — Physical images and OpenCV interoperability

- [x] SI-01: Optional installation and lazy imports; separate SimpleITK toolbox root.
- [x] SI-02: Typed native medical image connections with bounded 2D/3D real scalar/vector data.
- [x] SI-03: Single-file medical read and exclusive `.nii/.nii.gz/.mha/.nrrd` writes.
- [x] SI-04: Native-to-OpenCV converters with explicit XYZ/ZYX and slice/component/color choices.
- [x] SI-05: OpenCV-to-native reconstruction with preserved raw samples and explicit color interpretation.
- [x] SI-06: Exposed/remergeable JSON geometry and string tags, plus in-memory snapshot isolation.
- [x] SI-07: Physical crop/resize/pyramid/orthogonal-affine mapping; reject unsupported shear/projective geometry.
- [x] SI-08: Full processed-slice reinsertion on the exact original physical grid/source identity.
- [x] SI-09: Physical Gaussian smoothing and explicit intensity window/cast.
- [x] SI-10: Native binary/Otsu thresholding, including Otsu mask controls.
- [x] SI-11: Label Dice/Jaccard/counts with physical-grid checks and undefined-label handling.
- [x] SI-12: Public MRI/expert-label fixtures, source hashes/attribution, independent coordinate/count checks,
  saved projects and targeted GUI/headless validation.

Exit: raw samples, physical points, tickets and tags survive the qualified paths;
invalid combinations fail inside blocks. Examples are under
[`examples/SimpleITK Tools`](<../examples/SimpleITK Tools/README.md>). Native file
formats' metadata limitations are documented separately from in-memory retention.

## M2 — Acquisition, metadata persistence and volume review

- [ ] SI-13: Explicit DICOM series selection and GDCM ordering; per-slice tags,
  mixed/inconsistent spacing/orientation detection and series identity.
- [ ] SI-14: Lossless metadata sidecars bound to image hashes, conflict-safe paired
  publication/recovery and exact geometry/tag reload for restrictive file formats.
- [ ] SI-15: Finite volume/image sets, filename tickets, paired label/reference inputs
  and deterministic replay with per-case reports.
- [ ] SI-16: Physical index/point conversion blocks and typed physical coordinate frames.
- [ ] SI-17: Orthogonal/oblique multiplanar review with spacing-aware aspect ratios,
  slice navigation, window/level controls and label overlays.
- [ ] SI-18: Streaming/tiled volumes, work/memory budgets and explicit 4D/time/channel support.

Exit: a public multi-series or multi-volume dataset replays without changing patient,
series, slice or physical-frame associations; metadata exports remain bound to the
exact pixel files. DICOM export/de-identification needs its own explicit contract.

## M3 — Volumetric processing, segmentation and measurements

- [ ] SI-19: Reference-grid resampling with explicit transform direction,
  interpolation, default/background value and nearest-neighbor label handling.
- [ ] SI-20: N4 bias correction with masks, shrink factors and convergence controls.
- [ ] SI-21: Physical anisotropic/curvature diffusion, bilateral/median filtering
  and LoG/gradient/Laplacian tools with explicit spacing behavior.
- [ ] SI-22: 3D binary/grayscale morphology and connected-component/relabel workflows,
  including voxel radius vs physical radius and connectivity.
- [ ] SI-23: Region-growing/confidence-connected/connected-threshold/fast-marching
  seeds in index or physical coordinates, and geodesic/level-set segmentation.
- [ ] SI-24: Multi-Otsu, label selection, label voting/fusion, masking and label-safe composition.
- [ ] SI-25: Physical label-shape/intensity statistics, surfaces, centroids, bounding boxes,
  principal axes and per-label result collections.
- [ ] SI-26: Surface-distance/Hausdorff and boundary metrics, physical-volume errors,
  multi-label reports and explicit absent-label policies.

Exit: compare native calls and independent truth on labeled public cases; record
algorithm quality separately from interoperability and software correctness.
Reuse existing OpenCV/scikit-image compositions for suitable 2D tasks.

## M4 — Registration and reusable medical applications

- [ ] SI-27: Translation/rigid/similarity/affine registration, landmark initialization,
  reference/moving image direction, metrics, sampling and optimizer controls.
- [ ] SI-28: Multiresolution registration schedules and bounded training workers,
  progress/cancellation reporting and reproducible persisted transforms.
- [ ] SI-29: B-spline/deformable registration, displacement fields, Jacobian/quality
  diagnostics, inverse transforms where available and composite-transform persistence.
- [ ] SI-30: Apply trained transforms to images, labels, physical points and overlays
  with explicit source/reference frames and interpolation.
- [ ] SI-31: Public multimodal and longitudinal ground-truth/landmark example sets,
  with registration/segmentation evaluation reports.
- [ ] SI-32: Reusable ToolBlocks for medical batch processing and inspection;
  consider volumetric ML/DNN adapters only when a concrete workflow requires them.

Exit: reusable applications preserve spatial/ticket/model provenance through
save/reopen and replay, with measured quality and performance on the named datasets.
