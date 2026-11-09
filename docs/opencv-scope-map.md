# OpenCV image and video scope map

This map records where the OpenCV 5.0.0 module families fit in the
[image and video checklist](vision-coverage-checklist.md). It covers the
upstream core and contrib source module directories, the documented HAL group,
and the additional top-level namespaces exported by the installed OpenCV 5.0.0
contrib wheel. The [binding inventory](opencv-bindings.tsv) remains the
callable-level reference. The [OpenCV source module list](https://api.github.com/repos/opencv/opencv/contents/modules?ref=5.0.0)
and [contrib source module list](https://api.github.com/repos/opencv/opencv_contrib/contents/modules?ref=5.0.0)
anchor the module names. The [main](https://docs.opencv.org/5.0/main_modules/api_root.html)
and [extra](https://docs.opencv.org/5.0/extra_modules/api_root.html)
API references describe the families.

The IDs identify the graph capabilities owned by a family. A family can point
to several stages, and several families can support the same capability.
Count each capability once in its owning stage. A dash means the family has
no separate counted commitment. Rows with both a reference and a deferral
describe the part counted and the reason the remaining APIs are out of scope.
Build-dependent planned items need a verified distribution and a fixture before
they can count as complete.

## Main modules

| Module | Graph coverage and scope decisions | Checklist IDs |
| --- | --- | --- |
| `calib` | Camera targets, lens and planar maps, calibrated workplanes, stereo rig fitting and hand–eye geometry. Keep camera/robot transport with acquisition and industrial integration. | S1-69, S3-01–S3-09, S3-13–S3-14, S3-30–S3-32, S3-34–S3-41, S3-45, S3-54–S3-55, S3-70 |
| `core` | Pixel/channel operations, numeric image checks, spectral transforms, blending and synthetic noise. Defer general scalar, tensor, matrix and optimization APIs to internal implementation or user blocks. | S1-13–S1-18, S1-37, S1-41, S1-55, S1-70, S1-73, S1-78–S1-80, S1-96, S1-105, S1-119, S1-132, S1-135, S1-143 |
| `dnn` | Inference contracts, preprocessing, output adapters, model evaluation and learned vision. Keep importer internals and backend engines behind the model contract. | S2-01–S2-13, S2-23–S2-40, S2-45, S2-48, S2-51–S2-52 |
| `features` | Corner and descriptor pipelines, matching, affine-invariant features and review. Reuse a feature/matcher block for parameter variants with equivalent outputs. | S1-42–S1-44, S1-46, S1-64, S1-128, S2-14–S2-17, S2-44 |
| `flann` | Descriptor-based matching and image retrieval. Generic nearest-neighbor search structures stay behind feature adapters. | S1-44, S2-18–S2-19, S2-44 |
| `geometry` | 2D shape measurement and topology, robust image transforms, camera pose, triangulation and image-driven spatial fitting. | S1-30–S1-35, S1-38, S1-46–S1-48, S1-61, S1-63, S1-65, S1-99–S1-103, S1-131, S1-134, S1-136, S3-09–S3-12, S3-42–S3-44, S3-64–S3-65 |
| `highgui` | Windowing, mouse input, and interactive display belong to the existing Qt editor and viewer. | — |
| `imgcodecs` | Still images, buffers, TIFF pages, metadata, animation and precision-preserving codecs. The installed wheel reports JPEG 2000 enabled and AVIF and OpenEXR disabled. Qualify a supported build for additional codec variants. | S1-03–S1-08, S1-111, S1-125–S1-127, S3-49 |
| `imgproc` | Raster transforms, filtering, thresholds, masks, morphology, histograms, gradients, labels, overlays and image features. Drawing primitives support the overlay contract. | S1-13, S1-19–S1-30, S1-34–S1-36, S1-38–S1-40, S1-42, S1-45, S1-47, S1-56, S1-62, S1-66–S1-68, S1-71–S1-77, S1-93–S1-94, S1-97–S1-98, S1-106, S1-118, S1-120, S1-124, S1-129, S1-131–S1-133, S1-136–S1-137, S1-142–S1-144 |
| `java`, `js`, `objc`, `python` | Language bindings and packaging. Browser feasibility stays on the backend roadmap. | — |
| `objdetect` | Fiducials, code reading and generation, cascades and sliding-window detectors. Narrow recognition applications stay with job packages. | S1-60, S1-112–S1-114, S1-121–S1-122, S2-46–S2-47, S3-01, S3-10, S3-36–S3-37 |
| `photo` | Image restoration, exposure stacks, tone mapping, focus stacking, color transfer and photo filters. | S1-57–S1-59, S1-81–S1-92, S1-116, S1-138 |
| `ptcloud` | Camera-derived cloud transforms, visibility, normals, selection, local features, mesh and fusion. General-purpose point-cloud work stays outside the image roadmap. | S3-20–S3-27, S3-50–S3-53, S3-56–S3-60, S3-64, S3-71 |
| `stereo` | Calibrated and uncalibrated rectification, disparity, depth and comparison across baselines. | S3-13–S3-17, S3-44–S3-48, S3-61, S3-63, S3-66 |
| `stitching` | Registered mosaics, loop/overlap checks, panorama projection and seam blending. Internal stages share the same image/frame contracts. | S1-01, S1-49, S1-104, S1-116–S1-117, S1-124, S1-139 |
| `ts`, `world`, `HAL` | Native test support, monolithic packaging and hardware abstraction. Track qualified backend behavior separately. | — |
| `video` | Motion estimation, stateful image references, background subtraction, tracking, flow warps and temporal validation. | S1-02, S1-50–S1-54, S1-95, S1-107–S1-109, S1-130, S1-141, S2-28, S2-39–S2-40, S2-52 |
| `videoio` | Camera frames, video files and network streams, capture properties, playback, seeking and encoder qualification. Device SDKs and hardware sync stay on the acquisition roadmap. | S1-09–S1-12, S1-110, S1-115, S1-123, S1-140, S3-62 |

## Contrib modules

| Module | Graph coverage and scope decisions | Checklist IDs |
| --- | --- | --- |
| `alphamat` | Trimap matting with a supported Python build. | S4-60 |
| `bgsegm` | Adaptive background models and foreground sequence fixtures. | S4-21, S4-33, S4-56, S4-76 |
| `bioinspired` | Retina image and motion processing. Specialized biological modeling stays outside inspection scope. | S4-61 |
| `cannops`, `fastcv` | Vendor acceleration backends. Qualify the existing image workflow on the intended backend. | — |
| `ccalib` | Custom calibration targets and omnidirectional rigs. Check Python exposure on the intended build. | S3-05, S3-14, S3-54–S3-55, S3-70 |
| `cnn_3dobj` | Specialized research model and training interfaces. Camera object pose and learned image inference have separate graph contracts. | — |
| `cudaarithm`, `cudabgsegm`, `cudacodec`, `cudafeatures2d`, `cudafilters`, `cudaimgproc`, `cudalegacy`, `cudaobjdetect`, `cudaoptflow`, `cudastereo`, `cudawarping`, `cudev` | Accelerator/device variants of existing image and video families. The supported CPU path owns the capability. GPU parity belongs to backend qualification. | — |
| `cvv`, `ovis`, `plot`, `viz` | Debugging windows, graphing and native viewer APIs. The editor owns visual review and the reconstruction checklist covers camera-derived preview output. | — |
| `datasets`, `hdf` | Dataset loaders and general-purpose array storage. Treat fixtures and training asset formats as part of their consuming workflow. | — |
| `dnn_objdetect` | Object detection lives in the model-adapter contract. Defer standalone wrappers for retired, model-specific pipelines with no pinned fixture. | S2-08–S2-09, S2-25–S2-27 |
| `dnn_superres` | Learned single-image upscaling with a pinned model and measured reference. | S2-35 |
| `dnns_easily_fooled`, `dpm`, `face` | Defer specialized adversarial examples, legacy part-based detectors and face recognition. These need separate applications, model assets and acceptance data. | — |
| `freetype` | Unicode annotation on source images. The Qt renderer or another supported backend can supply glyphs on builds without native FreeType. | S1-144 |
| `fuzzy` | Fuzzy image smoothing and masked fill under the ft namespace. | S4-70 |
| `gapi` | Legacy processing remains available where supported. Future compiled graph execution belongs to the backend roadmap. | — |
| `hfs` | Hierarchical image segmentation and region output. | S4-71 |
| `img_hash` | Perceptual hashes with explicit descriptor and distance contracts. | S4-09–S4-10 |
| `intensity_transform` | Low-light enhancement. Shared gamma, log and linear intensity primitives belong to the baseline transform. | S1-17, S4-73 |
| `julia`, `matlab` | Language binding and platform packaging work. | — |
| `line_descriptor` | Line descriptors and matching with source-image endpoints. | S4-46 |
| `ml` | Classical model training, validation and scoring on image-derived features. Generic tabular estimators remain outside the roadmap. | S2-19–S2-23, S2-41–S2-44, S2-49–S2-51 |
| `optflow` | Dense and interpolated flow with known motion and consistency checks. | S4-22–S4-24, S4-57–S4-58, S4-77 |
| `phase_unwrapping` | Standalone unwrapping of measured fringe images. | S3-67 |
| `quality` | Full-reference error, similarity, gradient degradation and no-reference scoring. | S1-55, S1-119, S1-143, S4-07–S4-08, S4-32 |
| `rapid` | Metric object pose tracking against image silhouettes. | S3-68 |
| `reg` | Image pair registration with configurable mapper and residual checks. | S4-72 |
| `rgbd` | Typed depth and color, registration, pose tracking and depth fusion. Camera-frame acceptance is required before claiming a model or backend. | S3-18–S3-20, S3-23–S3-26, S3-49, S3-52–S3-53, S3-56–S3-57, S3-59, S3-62, S3-64, S3-66, S3-71 |
| `saliency` | Static and motion saliency maps with separate fixtures. | S4-53–S4-54 |
| `sfm` | Image-derived multi-view reconstruction and metric anchors. Qualify the relevant bindings or a tested solver. | S3-33, S3-58, S3-65 |
| `shape` | Shape context, Hausdorff distance and spline registration. | S4-50–S4-52 |
| `signal` | General one-dimensional signal primitives stay internal. Image frequency-domain capabilities use the core image transforms. | — |
| `structured_light` | Gray code, phase shift and calibrated projector/camera reconstruction. | S3-28–S3-30, S3-67 |
| `superres` | Classical multi-frame super-resolution. Qualify usable Python access before using the contrib backend. | S4-81 |
| `surface_matching` | Image-derived surface registration and alignment diagnostics, including ICP and point-pair features. | S3-22, S3-58 |
| `text` | Native text components and OCR adapters. Learned text models live in Stage 2. | S2-13, S2-31–S2-34, S4-67–S4-68 |
| `tracking` | Seeded and learned image tracking. Legacy trackers without a distinct target workflow are deferred. | S1-53–S1-54, S2-28, S2-40, S2-52 |
| `videostab` | Causal and look-ahead motion correction. Use the graph's bounded replay contracts and verify them across long clips. | S1-02, S1-50, S1-95 |
| `wechat_qrcode` | Small and low-contrast QR decoding with verified local assets. | S4-25 |
| `xfeatures2d` | Extended feature extraction and match filtering. Defer algorithms that lack a usable pinned build or transferable fixture until a specific workflow needs them. | S4-19–S4-20, S4-47–S4-49, S4-74–S4-75, S4-80 |
| `ximgproc` | Guided and adaptive filtering, superpixels, graph segmentation, thinning, line/ellipse detection and dense field interpolation. Depth-specific filters belong to Stage 3. | S3-16, S3-19, S4-01–S4-06, S4-13–S4-15, S4-18, S4-27–S4-31, S4-36–S4-45, S4-59, S4-69, S4-77–S4-79 |
| `xobjdetect` | HOG and configurable sliding-window detection. Legacy model-specific detectors stay under the existing model contracts. | S2-46, S4-26 |
| `xphoto` | White balance, denoising, tone mapping and selected restoration. | S4-12, S4-16–S4-17, S4-62–S4-66 |
| `xstereo` | Extra stereo correspondence models and confidence. Validate the desired variant on the pinned or explicitly supported wheel. | S3-16, S3-61, S3-72 |

## Additional wheel namespaces

These names supplement the source modules above. They include renamed and
nested families whose Python namespace differs from the C++ source module.
A namespace visible in the wheel can contain constants or API stubs. Confirm
callability and a representative fixture for each build-dependent commitment.

| Namespace | Graph coverage and scope decisions | Checklist IDs |
| --- | --- | --- |
| `aruco` | Marker and board generation/detection, camera pose and target validation. | S3-01, S3-10, S3-36–S3-38 |
| `barcode` | Native linear barcode reading. | S1-122 |
| `ccm`, `mcc` | Color chart detection, fitting and application. | S4-11, S4-34–S4-35 |
| `colored_kinfu`, `dynafu`, `kinfu`, `large_kinfu` | Depth tracking and fusion. Defer separate volume/backend variants until the bounded trajectory workflow has a fixture. | S3-25–S3-27, S3-71 |
| `cuda`, `ipp`, `ocl`, `parallel`, `videoio_registry` | Backend setup and capability reporting. Individual image workflows retain backend qualification criteria. | — |
| `detail` | Stitching internals and reusable Laplacian image pyramids. | S1-01, S1-49, S1-104, S1-117, S1-124 |
| `Error`, `data`, `instr`, `load_config_py3`, `mat_wrapper`, `misc`, `samples`, `typing`, `utils`, `version` | Python helpers, bindings, sample lookup, diagnostics and configuration. | — |
| `fisheye`, `omnidir`, `multicalib` | Lens models, camera geometry and rig fitting. | S3-04–S3-05, S3-09, S3-14, S3-54–S3-55, S3-70 |
| `ft` | Python namespace for fuzzy image transforms. | S4-70 |
| `legacy` | Compatibility wrappers for older interfaces and trackers. Cover new work under the current graph APIs. | — |
| `linemod` | Image/depth template matching for metric object pose. | S3-69 |
| `motempl` | Motion history images and motion direction. | S4-55 |
| `ogl` | Native OpenGL display interop. The application viewer owns display. | — |
| `ppf_match_3d` | Python namespace for point-pair surface matching. | S3-22, S3-58 |
| `segmentation` | Segmentation utilities. Keep implementations in their graph segmentation family. | S1-28, S1-97–S1-98, S4-04 |

## Adjacent image workflows

| Family | Graph coverage and scope decisions | Checklist IDs |
| --- | --- | --- |
| `Inspection measurements` | Image-space sampling, calipers, edge differences, usable pick support and curved-strip defects. Recipe delivery and production acceptance remain in Stage 5. | S1-35, S1-55–S1-56, S1-61, S1-63, S1-67–S1-68, S1-97, S1-101–S1-103, S1-119, S1-131, S1-134, S1-136–S1-137, S1-143, S3-07 |
| `Model and decoder adapters` | Explicit image workflows for LightGlue, Annoy and ZXing. General-purpose external libraries stay outside the denominator. | S1-114, S2-14–S2-18, S2-48–S2-50 |
| `Measurement fixtures` | Synthetic camera, video and model input variants are counted only when the workflow has a concrete fidelity check. | S1-105, S1-142, S2-23, S2-51, S3-48, S3-53, S4-76 |

## Maintain the baseline

When changing OpenCV versions, compare source modules and the binding inventory
against this map. Add or update dispositions for changed families. When adding a
roadmap capability, give it a stable ID and a measurable result, then add its
owning family reference before refreshing the coverage table. The
[roadmap](roadmap.md) gives implementation order. The checklist describes
graph coverage, while module mapping describes scope.
