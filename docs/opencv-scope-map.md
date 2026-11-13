# OpenCV image, video and point-cloud scope map

This map records where the OpenCV 5.0.0 module families fit in the
[coverage checklist](vision-coverage-checklist.md). It covers the
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
| `calib` | Camera targets, lens and planar maps, calibrated workplanes, stereo rig fitting and hand–eye geometry. Keep camera/robot transport with acquisition and industrial integration. Fit rigs with non-overlapping views as a separate workflow. | S1-69, S3-01–S3-09, S3-13–S3-14, S3-30–S3-32, S3-34–S3-41, S3-45, S3-54–S3-55, S3-70, S3-113, S3-127–S3-128 |
| `core` | Pixel/channel operations, numeric image checks, spectral transforms, blending and synthetic noise. Defer general scalar, tensor, matrix and optimization APIs to internal implementation or user blocks. | S1-13–S1-18, S1-37, S1-41, S1-55, S1-70, S1-73, S1-78–S1-80, S1-96, S1-105, S1-119, S1-132, S1-135, S1-143 |
| `dnn` | Inference contracts, preprocessing, output adapters, model evaluation and learned vision. Keep importer internals and backend engines behind the model contract. | S2-01–S2-13, S2-23–S2-40, S2-45, S2-48, S2-51–S2-52 |
| `features` | Corner and descriptor pipelines, matching, affine-invariant features and review. Reuse a feature/matcher block for parameter variants with equivalent outputs. | S1-42–S1-44, S1-46, S1-64, S1-128, S2-14–S2-17, S2-44 |
| `flann` | Descriptor-based matching and image retrieval. Generic nearest-neighbor search structures stay behind feature adapters. | S1-44, S2-18–S2-19, S2-44 |
| `geometry` | 2D shape measurement and topology, robust image transforms, camera pose, triangulation and image-driven spatial fitting. | S1-30–S1-35, S1-38, S1-46–S1-48, S1-61, S1-63, S1-65, S1-99–S1-103, S1-131, S1-134, S1-136, S3-09–S3-12, S3-42–S3-44, S3-64–S3-65, S1-174, S3-128–S3-131 |
| `highgui` | Windowing, mouse input, and interactive display belong to the existing Qt editor and viewer. | — |
| `imgcodecs` | Still images, buffers, TIFF pages, metadata, animation and precision-preserving codecs. The installed wheel reports JPEG 2000 enabled and AVIF and OpenEXR disabled. Qualify a supported build for additional codec variants. Verify codec probes and bounded page access. | S1-03–S1-08, S1-111, S1-125–S1-127, S1-149, S1-153, S1-158, S3-49 |
| `imgproc` | Raster transforms, filtering, thresholds, masks, morphology, histograms, gradients, labels, overlays and image features. Drawing primitives support the overlay contract. | S1-13, S1-19–S1-30, S1-34–S1-36, S1-38–S1-40, S1-42, S1-45, S1-47, S1-56, S1-62, S1-66–S1-68, S1-71–S1-77, S1-93–S1-94, S1-97–S1-98, S1-106, S1-118, S1-120, S1-124, S1-129, S1-131–S1-133, S1-136–S1-137, S1-142–S1-144 |
| `java`, `js`, `objc`, `python` | Language bindings and packaging. Browser feasibility stays on the backend roadmap. | — |
| `objdetect` | Fiducials, code reading and generation, cascades and sliding-window detectors. Narrow recognition applications stay with job packages. | S1-60, S1-112–S1-114, S1-121–S1-122, S2-46–S2-47, S3-01, S3-10, S3-36–S3-37 |
| `photo` | Image restoration, exposure stacks, tone mapping, focus stacking, color transfer and photo filters. | S1-57–S1-59, S1-81–S1-92, S1-116, S1-138 |
| `ptcloud` | Inspection cloud import, transforms, visibility, normals, sampling, segmentation, mesh and fusion. Vendor acquisition and unrelated GIS work have separate scope. | S3-20–S3-27, S3-50–S3-53, S3-56–S3-60, S3-64, S3-71 |
| `stereo` | Calibrated and uncalibrated rectification, disparity, depth and comparison across baselines. Verify three-camera rectification across two baselines. | S3-13–S3-17, S3-44–S3-48, S3-61, S3-63, S3-66, S3-114 |
| `stitching` | Registered mosaics, bounded tile rendering with reusable scan geometry, loop/overlap checks, panorama projection and seam blending. Internal stages share the same image/frame contracts. | S1-01, S1-49, S1-104, S1-116–S1-117, S1-124, S1-139 |
| `ts`, `world`, `HAL` | Native test support, monolithic packaging and hardware abstraction. Track qualified backend behavior separately. | — |
| `video` | Motion estimation, stateful image references, background subtraction, tracking, flow warps and temporal validation. Track flow interchange, pyramid reuse and seeded MIL tracking. | S1-02, S1-50–S1-54, S1-95, S1-107–S1-109, S1-130, S1-141, S2-28, S2-39–S2-40, S2-52, S1-154–S1-156, S1-170 |
| `videoio` | Camera frames, video files and network streams, capture properties, playback, seeking and encoder qualification. Device SDKs and hardware sync stay on the acquisition roadmap. Qualify bounded encoded input streams. | S1-09–S1-12, S1-110, S1-115, S1-123, S1-140, S3-62, S1-157 |

## Contrib modules

| Module | Graph coverage and scope decisions | Checklist IDs |
| --- | --- | --- |
| `alphamat` | Trimap matting with a supported Python build. | S4-60 |
| `bgsegm` | Adaptive background models, replay fixtures and restart qualification. | S4-21, S4-33, S4-56, S4-76, S1-170 |
| `bioinspired` | Retina image and motion processing. Specialized biological modeling stays outside inspection scope. | S4-61, S4-54, S4-98 |
| `cannops`, `fastcv` | Vendor acceleration backends. Qualify the existing image workflow on the intended backend. | — |
| `ccalib` | Custom calibration targets and omnidirectional rigs. Check Python exposure on the intended build. | S3-05, S3-14, S3-54–S3-55, S3-70 |
| `cnn_3dobj` | Specialized research model and training interfaces. Camera object pose and learned image inference have separate graph contracts. | — |
| `cudaarithm`, `cudabgsegm`, `cudacodec`, `cudafeatures2d`, `cudafilters`, `cudaimgproc`, `cudalegacy`, `cudaobjdetect`, `cudaoptflow`, `cudastereo`, `cudawarping`, `cudev` | Accelerator/device variants of existing image and video families. The supported CPU path owns the capability. GPU parity belongs to backend qualification. | — |
| `cvv`, `ovis`, `plot`, `viz` | Debugging windows, graphing and native viewer APIs. The editor owns visual review and the reconstruction checklist covers camera-derived preview output. | — |
| `datasets`, `hdf` | Dataset loaders and general-purpose array storage. Treat fixtures and training asset formats as part of their consuming workflow. | — |
| `dnn_objdetect` | Object detection lives in the model-adapter contract. Defer standalone wrappers for retired, model-specific pipelines with no pinned fixture. | S2-08–S2-09, S2-25–S2-27 |
| `dnn_superres` | Learned single-image upscaling with pinned models and single-scale or multi-scale output. | S2-35, S2-58 |
| `dnns_easily_fooled`, `dpm`, `face` | Defer specialized adversarial examples, legacy part-based detectors and face recognition. These need separate applications, model assets and acceptance data. | — |
| `freetype` | Unicode annotation on source images. The Qt renderer or another supported backend can supply glyphs on builds without native FreeType. | S1-144 |
| `fuzzy` | Fuzzy image smoothing and masked fill under the ft namespace. | S4-70, S4-100 |
| `gapi` | Legacy processing remains available where supported. Future compiled graph execution belongs to the backend roadmap. | — |
| `hfs` | Hierarchical image segmentation and region output. | S4-71 |
| `img_hash` | Perceptual hashes, score semantics and persisted image catalogs. | S4-09–S4-10, S4-96 |
| `intensity_transform` | Low-light enhancement. Shared gamma, log and linear intensity primitives belong to the baseline transform. | S1-17, S4-73 |
| `julia`, `matlab` | Language binding and platform packaging work. | — |
| `line_descriptor` | Line descriptors and matching with source-image endpoints. | S4-46, S4-99 |
| `ml` | Classical model training, validation and scoring on image-derived features. Generic tabular estimators remain outside the roadmap. | S2-19–S2-23, S2-41–S2-44, S2-49–S2-51, S2-59–S2-61 |
| `optflow` | Dense and interpolated flow with known motion and consistency checks. | S4-22–S4-24, S4-57–S4-58, S4-77 |
| `phase_unwrapping` | Standalone unwrapping of measured fringe images. | S3-67 |
| `quality` | Full-reference error, similarity, gradient degradation and no-reference scoring. | S1-55, S1-119, S1-143, S4-07–S4-08, S4-32 |
| `rapid` | Metric object pose tracking against image silhouettes. | S3-68 |
| `reg` | Image pair registration with configurable mapper and residual checks. | S4-72 |
| `rgbd` | Typed depth and color, registration, pose tracking and depth fusion. Camera-frame acceptance is required before claiming a model or backend. | S3-18–S3-20, S3-23–S3-26, S3-49, S3-52–S3-53, S3-56–S3-57, S3-59, S3-62, S3-64, S3-66, S3-71 |
| `saliency` | Static and motion maps and ranked objectness proposals with separate fixtures. | S4-53–S4-54, S4-97 |
| `sfm` | Image-derived multi-view reconstruction and metric anchors. Qualify the relevant bindings or a tested solver. | S3-33, S3-58, S3-65 |
| `shape` | Shape context, appearance-assisted ranking, Hausdorff distance and spline registration. | S4-50–S4-52, S4-95 |
| `signal` | Resample measured image profiles with explicit pitch and verified alias rejection. General signal utilities stay internal. | S1-173 |
| `structured_light` | Gray code, phase shift and calibrated projector/camera reconstruction. | S3-28–S3-30, S3-67 |
| `superres` | Classical multi-frame super-resolution. Qualify usable Python access before using the contrib backend. | S4-81 |
| `surface_matching` | Image-derived surface registration and alignment diagnostics, including ICP and point-pair features. | S3-22, S3-58 |
| `text` | Native text regions and OCR adapters. Learned text models live in Stage 2. | S2-13, S2-31–S2-34, S4-67–S4-68, S4-93–S4-94 |
| `tracking` | Seeded and learned image tracking. Legacy trackers without a distinct target workflow are deferred. | S1-53–S1-54, S2-28, S2-40, S2-52, S1-171 |
| `videostab` | Causal and look-ahead motion correction. Use the graph's bounded replay contracts and verify them across long clips. | S1-02, S1-50, S1-95 |
| `wechat_qrcode` | Small and low-contrast QR decoding with verified local assets. | S4-25 |
| `xfeatures2d` | Extended feature extraction and match filtering. Defer algorithms that lack a usable pinned build or transferable fixture until a specific workflow needs them. | S4-19–S4-20, S4-47–S4-49, S4-74–S4-75, S4-80 |
| `ximgproc` | Guided and adaptive filtering, superpixels, graph segmentation, thinning, line/ellipse detection and dense field interpolation. Depth-specific filters belong to Stage 3. | S3-16, S3-19, S4-01–S4-06, S4-13–S4-15, S4-18, S4-27–S4-31, S4-36–S4-45, S4-59, S4-69, S4-77–S4-79 |
| `xobjdetect` | HOG and configurable sliding-window detection. Legacy model-specific detectors stay under the existing model contracts. | S2-46, S4-26, S1-172, S2-61–S2-62 |
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
| `ccm`, `mcc` | Color chart detection, fitting, correction and independent profile verification. | S4-11, S4-34–S4-35, S4-101–S4-102 |
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
| `Inspection measurements` | Image-space sampling, calipers, edge differences, usable pick support and curved-strip defects. Recipe delivery and production acceptance remain in Stage 5. | S1-35, S1-55–S1-56, S1-61, S1-63, S1-67–S1-68, S1-97, S1-101–S1-103, S1-119, S1-131, S1-134, S1-136–S1-137, S1-143, S3-07, S1-174–S1-175, S3-127–S3-129 |
| `Model and decoder adapters` | Explicit image workflows for LightGlue, Annoy and ZXing. General-purpose external libraries stay outside the denominator. | S1-114, S2-14–S2-18, S2-48–S2-50 |
| `Measurement fixtures` | Synthetic camera, video and model input variants are counted only when the workflow has a concrete fidelity check. | S1-105, S1-142, S2-23, S2-51, S3-48, S3-53, S4-76 |

## Documented API subgroup audit

This pass reconciles 72 direct topics in the OpenCV 5.0 main algorithm
module pages and 35 in the contrib pages linked below. It also
separates point-cloud API families whose Python exposure differs from C++.
Rows group related topics that share one disposition. The parent module
tables cover modules with flat APIs and other contrib modules.

### Main algorithm subgroups

| Subgroup | Roadmap disposition | Checklist IDs |
| --- | --- | --- |
| `core / image structures and array operations` | Graph contracts cover typed pixel arrays and image operations. Scalar types, allocation, generic transforms and matrix algebra remain implementation details. | S1-13–S1-18, S1-41, S1-78–S1-80, S1-96 |
| `core / clustering` | Pixel clustering and image-derived codebooks are in scope. | S1-29, S2-19, S2-44 |
| `core / asynchronous and parallel execution` | Scheduling and task primitives belong to the graph runtime. | — |
| `core / persistence, utility and system APIs` | Application serialization owns profiles and assets. Native file-storage objects, diagnostics and system calls stay internal. | — |
| `core / GL, DirectX, OpenCL, VA-API and HAL` | Interoperability and acceleration belong to backend qualification. | — |
| `core / optimization and Eigen interoperability` | Keep generic solvers inside the measurement, calibration and inference paths that use them. | — |
| `core / quaternions` | Rotation data belongs in camera and object pose contracts. Defer generic quaternion utilities. | S3-09–S3-10, S3-31–S3-32 |
| `calib / fisheye` | Fit, project and rectify a declared lens model and qualify stereo separately. | S3-04, S3-09, S3-14, S3-55 |
| `features / detect and describe` | Standard corners and descriptors, stable regions and affine views. | S1-42–S1-43, S1-64, S1-128 |
| `features / matching, match graphics and approximate search` | Match descriptors, retain image coordinates and draw correspondences. Generic vector search stays behind the image-feature pipeline. | S1-44, S2-14–S2-19 |
| `features / hardware layer` | Keep accelerator variants under the existing algorithm contracts. | — |
| `geometry / shape fitting` | Fit, approximate, measure and compare image contours with reusable geometry. | S1-31–S1-35, S1-100, S1-131, S1-134, S1-136 |
| `geometry / camera and 3D geometry` | Estimate pose, epipolar geometry, triangulation and metric reconstruction. See the separate point-cloud geometry audit below. | S3-09–S3-12, S3-33, S3-42–S3-44, S3-65 |
| `imgcodecs / format flags` | Typed still and animated images, metadata, precision and codec qualification. | S1-03–S1-08, S1-111, S1-125–S1-127 |
| `imgcodecs / iOS and macOS glue` | Language and platform interoperability belong to packaging. | — |
| `imgproc / image filtering` | Smoothing, custom kernels, gradients, morphology, pyramids and frequency operations. | S1-22–S1-26, S1-40–S1-41, S1-66, S1-74–S1-80, S1-124, S1-129 |
| `imgproc / geometric and miscellaneous transforms` | Resize, remap, pad, threshold, distance, flood fill, pyramids, and frame-aware masks. | S1-25–S1-28, S1-36–S1-40, S1-69–S1-73, S1-93–S1-95, S1-118, S1-130, S1-142 |
| `imgproc / drawing` | Rasterize masks and annotated outputs. Primitive drawing calls stay behind the graphics and region contracts. | S1-62, S1-106, S1-132–S1-133, S1-144 |
| `imgproc / color conversion and palettes` | Typed color, sensor encodings and scalar display. | S1-08, S1-13–S1-14, S1-21, S1-96, S1-127 |
| `imgproc / planar subdivision` | Delaunay and Voronoi output on image coordinates. | S1-65 |
| `imgproc / histograms` | Selected histograms, comparison, backprojection and transport distance. | S1-19, S1-120 |
| `imgproc / contour structure and shape measures` | Extract topology, measurements and hulls. Feed them into region selection and contour fitting. | S1-30–S1-33, S1-97–S1-101, S1-131, S1-133–S1-134 |
| `imgproc / motion analysis and tracking` | Alignment, optical flow, temporal background and tracked regions. | S1-48, S1-50–S1-54, S1-107–S1-109, S1-130, S1-141 |
| `imgproc / feature and object detection` | Corners, tensors, lines, circles, shape and template detection. | S1-34, S1-42, S1-45–S1-47, S1-129 |
| `imgproc / image segmentation` | Thresholds, connected components, seeded foreground and labeled regions. | S1-25–S1-30, S1-97–S1-98 |
| `imgproc / hardware layer` | Track accelerator parity in backend qualification. | — |
| `objdetect / barcode and QR` | Native and optional decoder adapters, geometry, verification and QR encoding. Formal grading remains a separate application workflow. | S1-60, S1-112–S1-114, S1-121–S1-122, S4-25 |
| `objdetect / DNN face recognition` | Defer face-specific models and fixtures outside inspection scope. | — |
| `objdetect / common detector classes and fiducials` | Cascades, sliding-window detection, and marker or board geometry. | S2-46–S2-47, S3-01, S3-10, S3-36–S3-38 |
| `photo / inpainting and denoising` | Masked fills, single and temporal restoration, and controlled blur fixtures. | S1-57–S1-58, S1-84, S1-91–S1-92, S4-16–S4-17, S4-62 |
| `photo / HDR and decolorization` | Align brackets, fit response, merge, tone map and preserve contrast when removing color. | S1-59, S1-81–S1-83, S1-90, S4-63 |
| `photo / cloning and non-photorealistic rendering` | Seamless edits, photo smoothing, texture correction and sketching. | S1-85–S1-89 |
| `photo / chromatic aberration` | Qualify correction of known lateral color fringes. | S1-146 |
| `stitching / feature finding, matching and rotation estimation` | Baseline registered image mosaics and rotating-camera estimation. | S1-44, S1-49, S1-104, S1-147 |
| `stitching / autocalibration and warping` | Camera estimation and bounded tile rendering with frame and coverage checks. | S1-01, S1-38, S1-104, S1-147 |
| `stitching / seams, exposure and blending` | Qualified seam choice and exposure or multiband blending. | S1-49, S1-116–S1-117, S1-124 |
| `video / motion and object tracking` | Motion baseline, stabilization, foreground memory, flow warping, and track loss tests. | S1-02, S1-50–S1-54, S1-95, S1-107–S1-109, S1-130, S1-141, S2-40, S2-52 |
| `videoio / video flags, backend flags and registry` | File/network replay, seeking, camera property readback, timing and writer qualification. Registry enumeration is part of source qualification. | S1-09–S1-12, S1-110, S1-115, S1-123, S1-140 |
| `videoio / hardware codecs, iOS and WinRT glue` | Keep platform bindings and device-only acceleration outside the denominator. | — |
| `dnn / model layers and engine selection` | Verify model adapters with concrete images and report the engine used. Track native layer coverage through these fixtures. | S2-01–S2-13, S2-23–S2-40, S2-45, S2-48–S2-52 |
| `dnn / custom-layer registration` | Defer custom backend operators until a pinned image model requires them. | — |
| `ptcloud / RGB-D and volumetric reconstruction` | Depth conversion, camera-space registration, odometry and volumes. See the point-cloud audit below for individual families. | S3-18–S3-20, S3-25–S3-27, S3-71, S3-73–S3-76, S3-93–S3-95 |

Main topic indexes used for this audit include
[core](https://docs.opencv.org/5.0/main_modules/core.html#topics),
[calibration](https://docs.opencv.org/5.0/main_modules/calib.html#topics),
[features](https://docs.opencv.org/5.0/main_modules/features.html#topics),
[geometry](https://docs.opencv.org/5.0/main_modules/geometry.html#topics),
[image codecs](https://docs.opencv.org/5.0/main_modules/imgcodecs.html#topics),
[image processing](https://docs.opencv.org/5.0/main_modules/imgproc.html#topics),
[object detection](https://docs.opencv.org/5.0/main_modules/objdetect.html#topics),
[photo](https://docs.opencv.org/5.0/main_modules/photo.html#topics),
[stitching](https://docs.opencv.org/5.0/main_modules/stitching.html#topics),
[video](https://docs.opencv.org/5.0/main_modules/video.html#topics),
[video I/O](https://docs.opencv.org/5.0/main_modules/videoio.html#topics),
[DNN](https://docs.opencv.org/5.0/main_modules/dnn.html#topics), and
[point-cloud processing](https://docs.opencv.org/5.0/main_modules/ptcloud.html#topics).

### Contrib subgroups

| Subgroup | Roadmap disposition | Checklist IDs |
| --- | --- | --- |
| `ximgproc / structured edges, EdgeBoxes and line detectors` | Native edge, segment and proposal fixtures. | S4-18, S4-41–S4-45, S4-79 |
| `ximgproc / filters, superpixels and segmentation` | Guided and adaptive filtering, segmented label planes, and proposals. | S4-01–S4-06, S4-13–S4-15, S4-27–S4-31, S4-36–S4-40, S4-59, S4-69, S4-77–S4-78 |
| `ximgproc / contour spectra and run-length morphology` | Qualify spectral shape alignment and sparse binary-mask processing separately. | S4-82–S4-83 |
| `xfeatures2d / experimental descriptors and matching` | Verify additional descriptors and match filtering on registered image pairs. | S4-19–S4-20, S4-47–S4-49, S4-74–S4-75, S4-80 |
| `xfeatures2d / restricted features` | Keep algorithms that require a different build or licensing review deferred until an inspection fixture justifies qualification. | — |
| `xfeatures2d / image categorization` | Image-level descriptors and the model workflow own categorization. | S2-19–S2-23, S2-44 |
| `tracking / implementation and legacy interfaces` | Qualify active tracking behavior through graph blocks. Legacy tracker interfaces and internal model APIs stay deferred. | S1-53–S1-54, S2-40, S2-52 |
| `text / text detection and recognition` | Native candidate extraction, OCR adapter, and model-backed recognition with a pinned fixture. | S2-13, S2-31–S2-34, S4-67–S4-68 |
| `fuzzy / F0 and F1 primitives and image processing` | Track the reusable image filters. Keep the transform-specific math inside them. | S4-70 |
| `sfm / conditioning, epipolar fitting, projection and triangulation` | Review reprojection and metric reconstruction on known views. Keep numerical conditioning and solver primitives internal. | S3-09–S3-12, S3-33, S3-42–S3-44, S3-65 |
| `sfm / I/O, numeric utilities, robust fitting and reconstruction pipelines` | Keep generic numerics and data I/O inside the calibrated multi-view workflow. Validate the pinned backend or a tested solver with a known camera sequence. | S3-08, S3-33, S3-54, S3-65 |
| `videostab / global motion and fast-marching fill` | Qualify long-clip motion estimation and separate measured coverage from filled display pixels. | S1-02, S1-50, S1-95, S1-148 |
| `xobjdetect / cascade, HOG and extended detection` | Qualify generic image detector and descriptor contracts. Defer legacy detector internals and model-specific helper classes. | S2-46–S2-47, S4-26 |

Contrib topic indexes reviewed in this pass include
[extended image processing](https://docs.opencv.org/5.0/extra_modules/ximgproc.html#topics),
[extended features](https://docs.opencv.org/5.0/extra_modules/xfeatures2d.html#topics),
[tracking](https://docs.opencv.org/5.0/extra_modules/tracking.html#topics),
[fuzzy processing](https://docs.opencv.org/5.0/extra_modules/fuzzy.html#topics),
[structure from motion](https://docs.opencv.org/5.0/extra_modules/sfm.html#topics),
[stabilization](https://docs.opencv.org/5.0/extra_modules/videostab.html#topics),
[extended detection](https://docs.opencv.org/5.0/extra_modules/xobjdetect.html#topics), and
[scene text](https://docs.opencv.org/5.0/extra_modules/text.html#topics).

### Point-cloud and mesh API families

Presence in the installed wheel means the callable or class is exposed.
Each planned block still needs fixture validation. The C++ geometry
sampling, curvature and segmentation APIs lack direct bindings in this wheel.
Those rows require an explicitly qualified build or tested equivalent.

| Family | Roadmap disposition | Checklist IDs |
| --- | --- | --- |
| `Cloud import, export and attributes` | OpenCV exposes native PLY/OBJ cloud I/O. The app already supports PLY and ASCII PCD with its own codec. Extend interoperability for imported inspection scans and preserve explicitly declared attributes. | S3-21, S3-85, S3-96, S3-99 |
| `Spatial indexing and query` | The wheel exposes Octree construction, updates and nearest-neighbor searches. Put index construction and bounded query behind one cloud contract. | S3-77, S3-80–S3-81, S3-88 |
| `Cloud sampling and selection` | Keep the shipped graph bounds and voxel modes. Qualify representative, random and farthest-point modes separately. Preserve source indices and attributes. | S3-60, S3-78–S3-80 |
| `Cloud normals, curvature and segmentation` | Distinguish organized depth normals from unorganized cloud normals, robust geometric fits, clustering and normal-driven regions. | S3-20, S3-23, S3-51, S3-57, S3-81–S3-84 |
| `Cloud transforms, comparison and surface registration` | Separate alignment, overlap, hypothesis ranking, merged scans, dimensional deviation and physical bounds. | S3-22, S3-24, S3-50, S3-58, S3-88–S3-92, S3-98 |
| `Mesh I/O, projection and reconstruction` | Native mesh I/O and triangle rasterization are exposed in the wheel. Use separate mesh outputs for measured geometry and calibrated depth or color raster output. | S3-27, S3-52, S3-86–S3-87, S3-89, S3-97 |
| `Depth conversion and camera alignment` | Dense and sparse backprojection, unit normalization, ToF cleanup and RGB-D frame warp. | S3-17–S3-19, S3-49, S3-59, S3-62, S3-73–S3-76 |
| `Odometry, fusion and view synthesis` | OdometryFrame preprocessing and settings stay within the RGB-D contracts. Validate fixed, color and hashed volumes, volume growth, and explicitly posed raycasts. | S3-25–S3-27, S3-71, S3-93–S3-95 |
| `Scanner SDKs and specialized cloud formats` | Defer vendor acquisition, GIS/LAS workflows and native viewer backends. Recorded inspection-cloud imports and mesh outputs have their own fixtures. | — |

References for the family split include
[point-cloud processing](https://docs.opencv.org/5.0/main_modules/ptcloud.html),
[cloud sampling and segmentation](https://docs.opencv.org/5.0/main_modules/_3d.html),
[RGB-D and volumetric APIs](https://docs.opencv.org/5.0/main_modules/ptcloud.html#topics),
and [surface matching](https://docs.opencv.org/5.0/extra_modules/surface_matching.html).

### Native point-cloud API crosswalk

This crosswalk lists the function and class entry points in the OpenCV 5.0.0
point-cloud and depth headers, the 3D sampling and segmentation header, and
the installed wheel's PPF surface-matching namespace. Associated overloads,
settings and instance methods follow the graph contract in the same row.
An exposed Python API still needs a fixture before the capability is complete.

| API | Disposition on the pinned Python wheel | Checklist IDs |
| --- | --- | --- |
| `cv2.loadPointCloud`, `cv2.savePointCloud` | Exposed. Qualify native OBJ and compare PLY interoperability with the shipped graph codecs. | S3-21, S3-85, S3-96, S3-99 |
| `cv2.loadMesh`, `cv2.saveMesh` | Exposed. Mesh interchange owns index validation, supported topology and retained fields. | S3-86, S3-102, S3-107 |
| `cv2.triangleRasterize`, `cv2.triangleRasterizeColor`, `cv2.triangleRasterizeDepth`, `cv2.TriangleRasterizeSettings` | Exposed. Depth and color modes, shading, culling and camera conventions are configuration of one graph rasterizer. | S3-87, S3-102 |
| `cv2.Octree` | Exposed. Constructors, updates, extraction, and nearest and radius searches belong to the bounded spatial-index contract. The graph must keep stable scan indices when native queries return positions. | S3-77, S3-80–S3-81, S3-88, S3-103 |
| `cv2.RgbdNormals` | Exposed. Organized depth normals are shipped. Normal settings and invalid-sample handling belong to that graph tool. | S3-20 |
| `cv2.depthTo3d`, `cv2.depthTo3dSparse`, `cv2.rescaleDepth` | Exposed. Dense and sparse backprojection and typed depth conversion have independent contracts. | S3-59, S3-74–S3-75 |
| `cv2.registerDepth`, `cv2.warpFrame` | Exposed. Align depth between cameras or warp a registered depth and color pair through a pose. | S3-18, S3-76 |
| `cv2.findPlanes` | Exposed. Qualify multiple organized plane labels and coefficients separately from dominant-plane foreground separation. | S3-23, S3-100 |
| `cv2.Odometry`, `cv2.OdometryFrame`, `cv2.OdometrySettings` | Exposed. Frame preparation, pyramids, settings and normals remain part of odometry. Qualify trajectory recovery on sequences separately from the shipped pairwise fit. | S3-25, S3-71 |
| `cv2.Volume`, `cv2.VolumeSettings` | Exposed. Frame integration, color and depth, extraction, raycast and growth are grouped by graph workflow. Native volume state remains internal. | S3-26–S3-27, S3-93–S3-95 |
| `cv::randomSampling`, `cv::farthestPointSampling`, `cv::voxelGridSampling` | C++ functions with no direct binding in the pinned wheel. Qualify a supported build or an equivalent sampler with point provenance. | S3-60, S3-78–S3-79 |
| `cv::normalEstimate`, `cv::SACSegmentation`, `cv::RegionGrowing3D` | C++ APIs with no direct binding in the pinned wheel. Qualify fallback curvature, robust geometric fits and normal-based segmentation on explicit fixtures. | S3-82–S3-84 |
| `cv2.ppf_match_3d.ICP`, `cv2.ppf_match_3d.PPF3DDetector`, `cv2.ppf_match_3d.Pose3D`, `cv2.ppf_match_3d.PoseCluster3D` | Exposed. Training, matching, refinement, votes and pose clusters belong to the graph surface-matching contract. Raw pose object mutation stays internal. | S3-22, S3-90–S3-91 |
| `cv2.ppf_match_3d.computeNormalsPC3d`, `cv2.ppf_match_3d.samplePCByQuantization`, `cv2.ppf_match_3d.transformPCPose` | Exposed. The app already has typed normal, voxel and transform operations. Keep equivalent native variants in those contracts. | S3-50–S3-51, S3-60, S3-79, S3-84 |
| `cv2.ppf_match_3d.loadPLYSimple`, `cv2.ppf_match_3d.writePLY`, `cv2.ppf_match_3d.writePLYVisibleNormals` | Exposed. Reuse cloud interchange for PLY. Normal-debug rendering helpers belong to review graphics. | S3-21, S3-85 |
| `cv2.ppf_match_3d.addNoisePC`, `cv2.ppf_match_3d.getRandomPose` | Exposed. Keep native random helpers under a reproducible synthetic scan and pose fixture contract. | S3-111 |

Sources for the crosswalk include the pinned
[point-cloud header](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/ptcloud/include/opencv2/ptcloud.hpp),
[depth header](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/ptcloud/include/opencv2/ptcloud/depth.hpp),
[volume header](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/ptcloud/include/opencv2/ptcloud/volume.hpp),
[segmentation header](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/geometry/include/opencv2/geometry/segment.hpp),
and [surface-matching reference](https://docs.opencv.org/5.0/extra_modules/surface_matching.html).
The installed Python surface is checked against the [binding inventory](opencv-bindings.tsv).

### Native imaging, calibration and video API crosswalk

This crosswalk assigns named public functions and classes from the six pinned
headers below to graph capabilities. Overloads, option flags, and instance
methods follow their owning workflow. Python exposure and graph completion
are tracked separately. The audit covers 115 names, including 110 exposed in
Python and five C++ entries without a direct binding.

| API | Disposition on the pinned Python wheel | Checklist IDs |
| --- | --- | --- |
| `cv2.imread`, `cv2.imwrite` | Exposed. File and memory codecs share lossless PNG/JP2 and precision-aware TIFF/HDR controls. S1-126 qualifies uint16 sample/order preservation and codec/type/size checks. S1-125 native EXR qualification remains open. | S1-03, S1-125–S1-126 |
| `cv2.imdecode`, `cv2.imencode` | Exposed. PNG/JPEG, lossless JP2 and floating TIFF/HDR buffers use the typed packet contract and retain wire payloads. PNG filters/buffers and explicit JP2 coding mode are exposed. EXR remains capability guarded. | S1-04, S1-125–S1-126 |
| `cv2.imreadmulti`, `cv2.imwritemulti`, `cv2.imdecodemulti`, `cv2.imencodemulti` | Exposed. File replay and memory ports have separate acceptance checks. | S1-05, S1-149 |
| `cv2.imreadWithMetadata`, `cv2.imdecodeWithMetadata`, `cv2.imwriteWithMetadata`, `cv2.imencodeWithMetadata` | Qualified file/buffer baseline. PNG EXIF/XMP/ICC/CICP and JPEG EXIF/XMP/ICC are verified on readback; bounded canonical chunks, native framing, no-overwrite file publishing, all EXIF orientations, source coordinates and explicit fresh metadata after transformation. Opaque camera tags are not generally edited. File loading uses the buffer path for Unicode filenames. | S1-06–S1-07 |
| `cv2.Animation`, `cv2.imreadanimation`, `cv2.imwriteanimation`, `cv2.imdecodeanimation`, `cv2.imencodeanimation` | Exposed. One animation contract owns timing, order and supported formats for both paths. | S1-111 |
| `cv2.haveImageReader`, `cv2.haveImageWriter`, `cv2.imcount` | Exposed. Probe source readability, encoder availability and page count before replay. | S1-153 |
| `cv::ImageCollection` | C++ class without a direct binding. Qualify random page access with native support or indexed decode. | S1-158 |
| `cv2.inpaint` | Exposed. Telea and Navier–Stokes follow the masked restoration contract. | S1-57 |
| `cv2.AlignExposures`, `cv2.AlignMTB`, `cv2.createAlignMTB` | Exposed. Align exposures and use valid overlap for subsequent HDR merge. | S1-59, S1-150 |
| `cv2.MergeExposures`, `cv2.MergeMertens`, `cv2.createMergeMertens` | Exposed. Mertens owns uncalibrated exposure fusion. The base class stays internal. | S1-81–S1-82 |
| `cv2.CalibrateCRF`, `cv2.CalibrateDebevec`, `cv2.CalibrateRobertson`, `cv2.createCalibrateDebevec`, `cv2.createCalibrateRobertson` | Exposed. Response fitting belongs with calibrated HDR reconstruction. | S1-82 |
| `cv2.MergeDebevec`, `cv2.MergeRobertson`, `cv2.createMergeDebevec`, `cv2.createMergeRobertson` | Exposed. Qualify both calibrated merge choices and motion-rejected input. | S1-82, S1-150 |
| `cv2.Tonemap`, `cv2.TonemapDrago`, `cv2.TonemapMantiuk`, `cv2.TonemapReinhard`, `cv2.createTonemap`, `cv2.createTonemapDrago`, `cv2.createTonemapMantiuk`, `cv2.createTonemapReinhard` | Exposed. Qualify HDR-to-display mapping across supported operators and controls. | S1-83 |
| `cv2.detailEnhance`, `cv2.edgePreservingFilter` | Exposed. Qualify appearance restoration with retained defect boundaries. | S1-85 |
| `cv2.pencilSketch`, `cv2.stylization` | Exposed. Use as diagnostic appearance transforms with an explicit output contract. | S1-86 |
| `cv2.seamlessClone` | Exposed. Masked cloning belongs to reference preparation and fixture generation. | S1-87 |
| `cv2.illuminationChange`, `cv2.textureFlattening` | Exposed. Masked photo edits qualify localized variation/texture reduction and boundary position; report contrast loss and retain exact unselected samples. | S1-88 |
| `cv2.colorChange`, `cv2.decolor` | Exposed. Colored-step reference qualifies masked color editing; equal-gray patches qualify contextual decolorization. Native collapsed mappings withhold processed outputs. | S1-89–S1-90 |
| `cv2.fastNlMeansDenoising`, `cv2.fastNlMeansDenoisingColored`, `cv2.fastNlMeansDenoisingMulti`, `cv2.fastNlMeansDenoisingColoredMulti`, `cv2.denoise_TVL1` | Exposed. Single-frame and temporal variants share restoration fixtures by algorithm. | S1-91–S1-92 |
| `cv2.correctChromaticAberration`, `cv2.loadChromaticAberrationParams` | Exposed. Native parameter loading follows the fringe-correction workflow. | S1-146 |
| `cv2.VideoCapture`, `cv2.VideoWriter` | Exposed. Test capture and replay separately from encoder round trips. Flags and backend selection are configuration. | S1-09–S1-12, S1-110, S1-115, S1-123, S1-140 |
| `cv2.IStreamReader` | Exposed. Qualify bounded encoded input on a specific backend. | S1-157 |
| `cv2.CamShift`, `cv2.meanShift` | Qualified baseline. Explicit histogram teaching, native controls, color/work-area filters, current-ticket geometry, loss gates and explicit reseed after expiry; independent translation/scale fixtures. Color-mass geometry is not physical metrology or guaranteed identity. | S1-108 |
| `cv2.DenseOpticalFlow`, `cv2.SparseOpticalFlow`, `cv2.DISOpticalFlow`, `cv2.FarnebackOpticalFlow`, `cv2.SparsePyrLKOpticalFlow`, `cv2.calcOpticalFlowFarneback`, `cv2.calcOpticalFlowPyrLK` | Exposed. Algorithm objects and convenience calls share the same typed flow outputs. | S1-51 |
| `cv2.VariationalRefinement` | Qualified paired-image baseline. All native controls, independent motion/residual improvement, source/donor support, border exclusions, frame/ticket ownership and withheld-flow quality failures. Local single-scale refinement needs a suitable seed; diagnostic support does not prove visibility or accuracy. | S1-109 |
| `cv2.buildOpticalFlowPyramid` | Exposed. Qualify reusable Python pyramid/derivative representation, geometry and tracking parity. | S1-155 |
| `cv2.ECCParameters`, `cv2.findTransformECC`, `cv2.findTransformECCMultiScale`, `cv2.findTransformECCWithMask` | Exposed. The graph exposes single-scale, multiscale and masked alignment through the same contract. | S1-48 |
| `cv2.computeECC` | Exposed via `inspect.image_correlation`: selected aligned-image similarity, per-channel results and explicit constant-region policy. No transform is estimated. | S1-119 |
| `cv2.readOpticalFlow`, `cv2.writeOpticalFlow` | Exposed. Verify signed flow samples and frame association on import and export. | S1-154 |
| `cv2.KalmanFilter`, `cv2.Tracker`, `cv2.TrackerMIL` | Exposed. The generic interfaces support seeded tracking and prediction. MIL has its own clip fixture. | S1-53–S1-54, S1-156 |
| `cv2.TrackerDaSiamRPN`, `cv2.TrackerNano`, `cv2.TrackerVit` | Exposed. Qualify the model assets and lost-target behavior under learned tracking. | S2-40 |
| `cv::estimateRigidTransform` | Deprecated C++ function. Maintain the supported affine estimators in the existing alignment contract. | S1-48 |
| `cv2.calibrateCamera`, `cv2.calibrateCameraExtended`, `cv2.initCameraMatrix2D` | Exposed. Initial camera matrix and diagnostics belong to pinhole calibration. | S3-02 |
| `cv2.calibrateCameraRO`, `cv2.calibrateCameraROExtended` | Exposed. Qualify refined target geometry and its residuals. | S3-109 |
| `cv2.calibrateMultiview`, `cv2.calibrateMultiviewExtended` | Exposed. Shared target geometry and visibility belong to rig calibration. | S3-54 |
| `cv2.registerCameras`, `cv2.registerCamerasExtended` | Exposed. Verify registration with fixed intrinsics on non-overlapping target observations. | S3-113 |
| `cv2.stereoCalibrate`, `cv2.stereoCalibrateExtended`, `cv2.fisheye.calibrate`, `cv2.fisheye.stereoCalibrate` | Exposed. Track lens models and per-view stereo residuals in their owning workflows. | S3-04, S3-13–S3-14, S3-45 |
| `cv::calibrateHandEye`, `cv::calibrateRobotWorldHandEye` | C++ functions without direct bindings. Eye-in-hand has an offline graph solver. Joint robot-world and fixed-camera checks remain open. | S3-31–S3-32 |
| `cv2.stereoRectify`, `cv2.stereoRectifyUncalibrated`, `cv2.fisheye.stereoRectify` | Exposed. Qualify calibrated, uncalibrated and wide-angle image pairs separately. | S3-13–S3-14, S3-44 |
| `cv::rectify3Collinear` | C++ function without a direct binding. Qualify a three-camera rig across two baselines. | S3-114 |
| `cv2.StereoMatcher`, `cv2.StereoBM`, `cv2.StereoSGBM`, `cv2.validateDisparity` | Exposed. Matchers and consistency checks belong to disparity and confidence. | S3-15–S3-16 |
| `cv2.getValidDisparityROI`, `cv2.filterSpeckles`, `cv2.reprojectImageTo3D` | Exposed. Shared support, speckle removal and metric reprojection each have acceptance checks. | S3-17, S3-46–S3-47 |

Pinned OpenCV 5.0.0 sources include
[imgcodecs.hpp](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgcodecs/include/opencv2/imgcodecs.hpp),
[photo.hpp](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/photo/include/opencv2/photo.hpp),
[videoio.hpp](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/videoio/include/opencv2/videoio.hpp),
[tracking.hpp](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/video/include/opencv2/video/tracking.hpp),
[calib.hpp](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/calib/include/opencv2/calib.hpp), and
[stereo.hpp](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/stereo/include/opencv2/stereo.hpp).
Python names are checked against the [binding inventory](opencv-bindings.tsv).

### Native raster, feature, detection and camera geometry API crosswalk

This audit maps 254 top-level functions, aliases and classes from 14 pinned
headers. The Python wheel exposes 236 of these names. The other 18 are C++
entries without direct bindings. Nested parameter types, constructors,
flags and instance methods follow their owning workflow. A Python binding
is evidence of exposure. Completion requires the checklist's graph fixture.

#### Raster processing and annotation

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.GaussianBlur`, `cv2.getGaussianKernel` | Gaussian kernel generation is part of the existing smoothing tool contract. | S1-22 |
| `cv2.filter2D`, `cv2.filter2Dp`, `cv::Filter2DParams`, `cv2.sepFilter2D` | Convolution settings belong to the editable kernel. The parameter class has no direct binding in this wheel. | S1-23 |
| `cv2.Canny`, `cv2.Laplacian`, `cv2.Scharr`, `cv2.Sobel`, `cv2.spatialGradient`, `cv2.getDerivKernels` | Gradient kernels and filters belong to derivative, edge and focus tools. S1-138 now uses local Sobel/Laplacian energy for hard/weighted registered-stack fusion, with captured-neighborhood halos, original level indices, coverage and ambiguity fallback. Independent sharp-image and upstream ECC examples qualify the workflow. | S1-24, S1-138 |
| `cv2.blur`, `cv2.boxFilter`, `cv2.medianBlur`, `cv2.bilateralFilter` | Box, median and bilateral options have separate image fixtures. | S1-74–S1-76 |
| `cv2.getGaborKernel` | Generate oriented texture kernels in the existing Gabor response workflow. | S1-77 |
| `cv2.stackBlur` | Qualify large-radius smoothing and edge retention. | S1-159 |
| `cv2.integral`, `cv2.integral2`, `cv2.integral3`, `cv2.sqrBoxFilter` | Sum, squared sum and tilted sum support local image statistics. | S1-20 |
| `cv2.cvtColor`, `cv2.cvtColorTwoPlane`, `cv2.demosaicing` | Route sensor layouts and typed color conversions through existing contracts. `cv.decode_raw_sensor`/`input.raw_sensor` qualify MIPI RAW10/12 groups and strides; `cv.demosaic_bayer` optionally follows captured phase through integer pixel geometry. Other vendor packing and camera SDKs require separate adapters. | S1-08, S1-13, S1-127 |
| `cv2.CLAHE`, `cv2.createCLAHE`, `cv2.equalizeHist`, `cv2.applyColorMap` | Equalization and palettes use typed image inputs. | S1-21 |
| `cv2.calcHist`, `cv2.calcBackProject`, `cv2.compareHist`, `cv2.EMD`, `cv::wrapperEMD` | Histogram reduction, projection and transport distance. The C++ EMD wrapper stays internal. | S1-19, S1-120 |
| `cv2.blendLinear` | Use per-pixel weights within the blending contract. cv.alpha_over separately implements CPU source-over equations with straight/premultiplied association, typed storage and transparency checks against scalar rational truth; its alpha treatment is distinct from normalized two-weight blending. | S1-16, S1-132 |
| `cv2.threshold`, `cv2.thresholdWithMask`, `cv2.adaptiveThreshold` | Test thresholds, selections and boundary behavior in one family. | S1-25 |
| `cv2.dilate`, `cv2.erode`, `cv2.morphologyEx`, `cv2.getStructuringElement` | Group morphology by operator and kernel. Binary hit-or-miss has a separate fixture. | S1-26, S1-151 |
| `cv2.floodFill`, `cv2.distanceTransform`, `cv2.distanceTransformWithLabels`, `cv2.grabCut`, `cv2.watershed` | Seeds, segmentation and distances retain image coordinates. S1-136 now compares directed mask boundaries and raw/beyond-tolerance missing/extra area; source-grid query selections retain uncut target boundaries. Explicit metrics, neighborhood/exterior policies, empty-set validity and independent pixel truth qualify the comparison. | S1-27–S1-28, S1-98, S1-136 |
| `cv2.connectedComponents`, `cv2.connectedComponentsWithAlgorithm`, `cv2.connectedComponentsWithStats`, `cv2.connectedComponentsWithStatsWithAlgorithm`, `cv2.SimpleBlobDetector` | Algorithm variants follow the label and measurement contract. Check label consistency where order differs. | S1-30, S1-97 |
| `cv2.findContours`, `cv2.findContoursLinkRuns` | Verify contour hierarchy and compare sparse versus dense extraction. | S1-31, S1-152 |
| `cv2.pyrUp`, `cv2.pyrDown`, `cv::buildPyramid`, `cv2.pyrMeanShiftFiltering` | Pyramid helpers stay in Gaussian and Laplacian processing. Mean-shift smoothing retains its own contract. | S1-40, S1-66, S1-124 |
| `cv2.resize`, `cv2.getRectSubPix` | Distinguish resizing from fractional patch sampling and report source coordinates. | S1-36, S1-71 |
| `cv2.remap`, `cv2.convertMaps`, `cv2.warpPolar` | Use typed maps and report coverage after resampling. cv.warp_flow exposes pull/forward sampling, conservative support, inverse/round-trip evidence and source-ticket/target-grid lineage. cv.interpolate_frames preserves paired-source timing. Occlusion masks are labeled consistency proxies with known-motion and missing-support checks. | S1-39, S1-93–S1-95, S1-130 |
| `cv2.getAffineTransform`, `cv2.getPerspectiveTransform`, `cv2.invertAffineTransform`, `cv2.warpAffine`, `cv2.warpPerspective`, `cv2.getRotationMatrix2D`, `cv::getRotationMatrix2D_` | Inverse and matrix representation choices stay inside the graph warp contract. | S1-38, S1-72, S3-06 |
| `cv2.arrowedLine`, `cv2.circle`, `cv2.clipLine`, `cv2.drawContours`, `cv2.drawFrameAxes`, `cv2.drawMarker`, `cv2.ellipse`, `cv2.ellipse2Poly`, `cv2.fillConvexPoly`, `cv2.fillPoly`, `cv2.line`, `cv2.polylines`, `cv2.rectangle`, `cv::LineIterator` | collection.rasterize_contours groups related subtree boundaries and unions independent roots, with explicit solid/even-odd alternatives, binary filled/outline sampling, fixed-point controls and source-grid context. Other drawing and C++ pixel traversal stay internal to selections and result graphics. | S1-62, S1-106, S1-133, S3-10 |
| `cv2.FontFace`, `cv2.getFontScaleFromHeight`, `cv2.getTextSize`, `cv2.putText` | Text styles and sizing stay in annotation. Qualify native glyph coverage for Unicode. | S1-106, S1-144 |
| `cv2.BackgroundSubtractor`, `cv2.BackgroundSubtractorKNN`, `cv2.BackgroundSubtractorMOG2`, `cv2.createBackgroundSubtractorKNN`, `cv2.createBackgroundSubtractorMOG2` | Use the existing stateful warmup and reset contract. | S1-52 |
| `cv2.accumulate`, `cv2.accumulateProduct`, `cv2.accumulateSquare`, `cv2.accumulateWeighted` | S1-137 now builds bright/dark references and optional population temporal-noise maps using bounded, scaled NumPy mean/median reductions; it does not claim these individual bindings. Per-pixel covariance and general reference learning remain separate workflows. | S1-107, S1-137, S1-160 |

#### Features, matching and 2D geometry

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.cornerHarris`, `cv2.cornerMinEigenVal`, `cv2.cornerSubPix`, `cv2.preCornerDetect`, `cv2.GFTTDetector`, `cv2.goodFeaturesToTrack`, `cv2.goodFeaturesToTrackWithQuality`, `cv2.FastFeatureDetector`, `cv::FAST` | Corner scores and subpixel coordinates are existing output families. Use the detector class for FAST. | S1-42 |
| `cv2.cornerEigenValsAndVecs` | inspect.structure_tensor supplies sorted eigenpairs, normal/tangent orientation, coherence, validity and an axial summary with Sobel/Scharr and border controls. Analytic ramps and a 16-bit six-probe fixture verify orientations, ambiguous corners, flat/weak signals and source-frame selection. | S1-129 |
| `cv2.LineSegmentDetector`, `cv2.createLineSegmentDetector`, `cv2.HoughCircles`, `cv2.HoughLines`, `cv2.HoughLinesP`, `cv2.HoughLinesPointSet` | Detected geometry uses existing line, segment and circle contracts. | S1-34 |
| `cv2.GeneralizedHough`, `cv2.GeneralizedHoughBallard`, `cv2.GeneralizedHoughGuil`, `cv2.createGeneralizedHoughBallard`, `cv2.createGeneralizedHoughGuil` | Two variants share image-space pose, votes and result graphics. | S1-47 |
| `cv2.matchTemplate` | Use the existing multipeak template detection workflow. | S1-45 |
| `cv2.createHanningWindow`, `cv2.phaseCorrelate`, `cv2.phaseCorrelateIterative` | Use the window internally in spectral alignment. S1-139 now combines Fourier magnitude and log-polar proposals with translation, spatial similarity verification and distinct-pose ambiguity gates; original-grid controls qualify its inverse geometry. | S1-48, S1-80, S1-139 |
| `cv2.Feature2D`, `cv2.ORB`, `cv2.SIFT`, `cv::KeyPointsFilter` | Feature records own keypoint filtering, descriptors and source frame. | S1-42–S1-43, S1-63 |
| `cv2.MSER`, `cv2.AffineFeature` | Stable regions have separate output checks. inspect.affine_features uses native AffineFeature with SIFT/ORB, explicit bounded schedules, source-space support geometry and a known-tilt correspondence test. Contrib AffineFeature2D variants remain separate build-dependent candidates. | S1-64, S1-128 |
| `cv2.DescriptorMatcher`, `cv2.BFMatcher`, `cv2.FlannBasedMatcher`, `cv::SL2`, `cv::Accumulator` | Match modes share the descriptor contract. Distance and accumulation types stay internal. | S1-44 |
| `cv2.drawKeypoints`, `cv2.drawMatches`, `cv2.drawMatchesKnn` | Use correspondence and keypoint graphics without changing the typed feature results. | S1-44, S1-106 |
| `cv::evaluateFeatureDetector`, `cv::computeRecallPrecisionCurve`, `cv::getRecall`, `cv::getNearestPoint` | Use a tested equivalent for image-pair benchmarking. These C++ functions lack direct Python bindings. | S1-163 |
| `cv2.DISK`, `cv2.ALIKED`, `cv2.LightGlueMatcher`, `cv2.ANNIndex` | Existing workflows own learned features and embedding retrieval. Qualify native ONNX LightGlue and the native embedding index separately. | S2-14, S2-16–S2-18, S2-53–S2-54 |
| `cv2.approxPolyDP`, `cv2.approxPolyN`, `cv2.arcLength`, `cv2.boundingRect`, `cv2.contourArea` | Contour simplification and measurements preserve source indices and hierarchy. | S1-30–S1-31, S1-99 |
| `cv2.HuMoments`, `cv2.moments`, `cv2.convexHull`, `cv2.convexityDefects`, `cv2.isContourConvex`, `cv2.matchShapes`, `cv2.pointPolygonTest` | Reuse contour and image-weighted measurements. inspect.intensity_moments adds selected contrast/binary centers, stable covariance/axes, explicit working-weight scale and image_pose_v1 output, with analytic rectangle-mixture and tiny/HDR checks. | S1-32, S1-131 |
| `cv2.boxPoints`, `cv2.fitEllipse`, `cv2.fitEllipseAMS`, `cv2.fitEllipseDirect`, `cv2.fitLine`, `cv2.getClosestEllipsePoints`, `cv2.intersectConvexConvex`, `cv2.minAreaRect`, `cv2.minEnclosingCircle`, `cv2.rotatedRectangleIntersection` | Fitted geometry, residuals, and overlap follow the existing measurements. | S1-33, S3-07 |
| `cv2.minEnclosingConvexPolygon`, `cv2.minEnclosingTriangle`, `cv2.Subdiv2D` | Distinguish minimum convex enclosures from image-space Delaunay and Voronoi outputs. | S1-65, S1-100 |

#### Codes, camera targets and geometry

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.GraphicalCodeDetector`, `cv2.QRCodeDetector`, `cv2.QRCodeDetectorAruco`, `cv2.QRCodeEncoder`, `cv2.barcode.BarcodeDetector` | Use existing code records for detection and payload. Qualify the ArUco-backed QR detector and ordered structured-append payload separately. | S1-60, S1-112, S1-121–S1-122, S1-161–S1-162 |
| `cv2.QRCodeEncoder_create`, `cv2.QRCodeEncoder.encode`, `cv2.QRCodeEncoder.encodeStructuredAppend` | Qualified generation baseline. All native parameters, native/independent binary references, UTF-8/validated Shift-JIS, explicit quiet zones, bounded sheets and grouped symbols. Native ASCII/UTF-8 recovery and optional ZXing Kanji recovery are checked; structured-append ordering/recovery and grading remain separate. | S1-112, S1-162 |
| `cv2.FaceDetectorYN`, `cv2.FaceRecognizerSF` | Face-specific model pipelines are deferred. Generic inference and detection belong to Stage 2. | — |
| `cv2.aruco.ArucoDetector`, `cv2.aruco.DetectorParameters`, `cv2.aruco.RefineParameters`, `cv2.aruco.drawDetectedMarkers`, `cv2.aruco.getPredefinedDictionary` | Retain single marker results. Known-board recovery owns rejected candidate refinement and board correspondences. | S3-01, S3-120 |
| `cv2.aruco.Board`, `cv2.aruco.GridBoard`, `cv2.aruco.CharucoBoard`, `cv2.aruco.CharucoDetector`, `cv2.aruco.CharucoParameters`, `cv2.aruco.drawDetectedCornersCharuco`, `cv2.aruco.drawDetectedDiamonds` | Group board generation, capture and pose inputs. Drawing is part of review graphics. | S3-36–S3-38, S3-120 |
| `cv2.aruco.Dictionary`, `cv2.aruco.extendDictionary`, `cv2.aruco.generateImageMarker` | Fixed target printing and custom dictionary qualification have distinct acceptance checks. | S3-37, S3-119 |
| `cv2.mcc.CChecker`, `cv2.mcc.CCheckerDetector`, `cv2.mcc.DetectorParametersMCC` | Chart detection keeps sample positions, chart type and review graphics. | S4-34 |
| `cv2.CirclesGridFinderParameters`, `cv2.findCirclesGrid` | Grid topology and clustering options belong to calibration target detection. | S3-35 |
| `cv2.checkChessboard`, `cv2.drawChessboardCorners`, `cv2.estimateChessboardSharpness`, `cv2.find4QuadCornerSubpix`, `cv2.findChessboardCorners`, `cv2.findChessboardCornersSB`, `cv2.findChessboardCornersSBWithMeta` | Check capture quality and refine corners. Oversized patterns need metadata-aware qualification. | S3-34, S3-110 |
| `cv2.getDefaultNewCameraMatrix`, `cv2.getOptimalNewCameraMatrix`, `cv::getUndistortRectangles`, `cv2.initUndistortRectifyMap`, `cv2.undistort`, `cv::undistortImage` | Camera grid, valid crop and output intrinsics belong to rectification. | S1-69, S3-03, S3-40 |
| `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify`, `cv::initWideAngleProjMap` | Wide-angle projection follows the calibrated lens contract. Qualify an equivalent map for the C++-only initializer when needed. | S3-04–S3-05, S3-55 |
| `cv2.initInverseRectificationMap` | Projector images need the inverse map with their own directional round-trip fixture. | S3-116 |
| `cv2.fisheye.distortPoints`, `cv2.fisheye.projectPoints`, `cv2.fisheye.undistortPoints`, `cv2.projectPoints`, `cv2.projectPointsSepJ`, `cv2.undistortImagePoints`, `cv2.undistortPoints` | Lens model, point direction and distortion belong to calibrated projection. Jacobian details stay internal. | S3-09 |
| `cv2.Rodrigues`, `cv2.solveP3P`, `cv2.solvePnP`, `cv2.solvePnPGeneric`, `cv2.solvePnPRansac`, `cv2.solvePnPRefineLM`, `cv2.solvePnPRefineVVS` | Pinhole solver variants belong to pose estimation and explicit pose hypotheses. | S3-10, S3-42 |
| `cv2.fisheye.solvePnP`, `cv2.fisheye.solvePnPRansac` | These native solvers are available in the pinned wheel. The graph object-pose block still requires a pinhole camera; qualify wide-angle pose and source-lens reprojection. | S3-130 |
| `cv2.UsacParams`, `cv2.decomposeEssentialMat`, `cv2.findEssentialMat`, `cv2.findFundamentalMat`, `cv2.recoverPose`, `cv2.computeCorrespondEpilines` | Keep robust solver settings with epipolar fitting and relative camera pose. The shipped relative-pose path uses one shared pinhole camera; independent intrinsics and lens models need qualification. | S3-11, S3-33, S3-131 |
| `cv2.triangulatePoints`, `cv2.convertPointsFromHomogeneous`, `cv2.convertPointsToHomogeneous`, `cv::convertPointsHomogeneous` | Coordinate normalization belongs inside calibrated projection and triangulation. | S3-12, S3-65 |
| `cv2.correctMatches`, `cv2.sampsonDistance` | Keep correspondence adjustment and error in the epipolar quality contract. | S3-43 |
| `cv2.estimateAffine2D`, `cv2.estimateAffinePartial2D`, `cv2.estimateTranslation2D`, `cv2.findHomography` | Use typed inliers and geometry for reference alignment and planar mapping. | S1-48, S3-06 |
| `cv2.estimateAffine3D`, `cv2.estimateTranslation3D`, `cv2.composeRT` | Keep relative pose composition and metric landmark transforms with their owning workflows. | S3-31, S3-50, S3-101 |
| `cv2.calibrationMatrixValues`, `cv2.decomposeProjectionMatrix`, `cv2.RQDecomp3x3` | Recover an interpretable camera profile. Keep raw factorization internal. | S3-117 |
| `cv2.decomposeHomographyMat`, `cv2.filterHomographyDecompByVisibleRefpoints` | Recover image-to-image motion hypotheses and reject candidates with invalid visibility. | S3-118 |
| `cv::LevMarq`, `cv2.matMulDeriv` | Solver and matrix-derivative internals stay behind calibrated image workflows. | — |

Pinned sources include the headers for
[image processing](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgproc/include/opencv2/imgproc.hpp),
[features](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/features/include/opencv2/features.hpp),
[2D geometry](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/geometry/include/opencv2/geometry/2d.hpp),
[3D geometry](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/geometry/include/opencv2/geometry/3d.hpp),
[object detection](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/objdetect/include/opencv2/objdetect.hpp) and its eight detector and target headers,
and [background subtraction](https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/video/include/opencv2/video/background_segm.hpp).

### Native DNN and stitching API crosswalk

This audit maps 151 named declarations from the DNN interface header and
12 stitching headers. The wheel exposes 89 of these names. It also exposes
instance methods such as DNN per-layer profiling, which follow their owning
graph item. Engine, importer, warper and accelerator implementation variants
stay with the corresponding inspection workflow.

#### Model loading, image adapters and inference

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.dnn.Net`, `cv2.dnn.readNetFromONNX`, `cv2.dnn.getAvailableTargets`, `cv::dnn::getAvailableBackends` | Keep engine selection, named inputs and outputs, profiling and quantized model qualification in the model contract. | S2-01–S2-05, S2-55, S2-57 |
| `cv2.dnn.readNet`, `cv2.dnn.readNetFromModelOptimizer`, `cv2.dnn.readNetFromTFLite`, `cv2.dnn.readNetFromTensorflow`, `cv2.dnn.readTensorFromONNX`, `cv2.dnn.writeTextGraph`, `cv::dnn::modelFormatToString` | Qualify additional import formats only for a pinned inspection model. Reference tensor loading can support reproducible fixtures. Legacy conversion helpers stay internal. | S2-01, S2-03, S2-05 |
| `cv2.dnn.Image2BlobParams`, `cv2.dnn.blobFromImage`, `cv2.dnn.blobFromImageWithParams`, `cv2.dnn.blobFromImages`, `cv2.dnn.blobFromImagesWithParams`, `cv2.dnn.imagesFromBlob` | Keep typed model preprocessing, per-image batches and image/tensor coordinate maps together. | S2-04–S2-05, S2-24 |
| `cv2.dnn.Model`, `cv2.dnn.ClassificationModel`, `cv2.dnn.DetectionModel`, `cv2.dnn.KeypointsModel`, `cv2.dnn.SegmentationModel` | Model convenience classes share typed graph decoders, confidence and source-coordinate mapping. | S2-06–S2-12, S2-25–S2-26, S2-45 |
| `cv2.dnn.NMSBoxes`, `cv2.dnn.NMSBoxesBatched`, `cv2.dnn.NMSBoxesRotated`, `cv2.dnn.softNMSBoxes` | Select suppression by box type and class policy. Score decay has a separate fixture and result. | S2-08–S2-09, S2-27, S2-56 |
| `cv2.dnn.TextDetectionModel`, `cv2.dnn.TextDetectionModel_DB`, `cv2.dnn.TextDetectionModel_EAST`, `cv2.dnn.TextRecognitionModel`, `cv2.dnn.Tokenizer` | DB and EAST follow the text region contract. Use a tokenizer only where a pinned image-to-text model requires it. | S2-13, S2-31–S2-34 |
| `cv::dnn::ActivationLayer`, `cv::dnn::Arg`, `cv::dnn::ArgData`, `cv::dnn::Graph`, `cv2.dnn.Layer`, `cv::dnn::LayerParams`, `cv::dnn::argKindToString`, `cv::dnn::enableModelDiagnostics` | Layer, graph, argument and importer internals stay behind model fixtures. Keep diagnostic import details inside the runtime. | — |

#### Registration, mosaics and registered replay

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.Stitcher` | Use configurable scan or panorama workflows. Qualify composition with reusable geometry and preserve camera and tile provenance. | S1-01, S1-49, S1-104, S1-164 |
| `cv2.detail.AffineBestOf2NearestMatcher`, `cv2.detail.BestOf2NearestMatcher`, `cv2.detail.BestOf2NearestRangeMatcher`, `cv2.detail.FeaturesMatcher`, `cv2.detail.ImageFeatures`, `cv2.detail.LightGlueFeaturesMatcher`, `cv2.detail.MatchesInfo`, `cv2.detail.computeImageFeatures`, `cv2.detail.computeImageFeatures2` | Use native matchers as implementation choices and preserve confidence, inliers and disconnected views. | S1-44, S1-49, S1-104, S1-147, S2-53 |
| `cv2.detail.calibrateRotatingCamera`, `cv::detail::estimateFocal`, `cv2.detail.focalsFromHomography` | Estimate rotating-camera intrinsics and report residuals on held-out overlaps. | S1-147 |
| `cv2.detail.AffineBasedEstimator`, `cv2.detail.BundleAdjusterAffine`, `cv2.detail.BundleAdjusterAffinePartial`, `cv2.detail.BundleAdjusterBase`, `cv2.detail.BundleAdjusterRay`, `cv2.detail.BundleAdjusterReproj`, `cv2.detail.Estimator`, `cv2.detail.HomographyBasedEstimator`, `cv2.detail.NoBundleAdjuster`, `cv::detail::autoDetectWaveCorrectKind`, `cv::detail::findMaxSpanningTree`, `cv2.detail.leaveBiggestComponent`, `cv2.detail.matchesGraphAsString`, `cv2.detail.waveCorrect` | Separate affine scan fitting from rotating-camera fits. Keep solver, match-graph and wave correction choices in their registration contract. | S1-49, S1-104, S1-147 |
| `cv2.detail.CameraParams` | Persist the calibrated camera or view layout with a declared output grid. | S1-147, S1-164 |
| `cv2.detail.BlocksChannelsCompensator`, `cv2.detail.BlocksCompensator`, `cv2.detail.BlocksGainCompensator`, `cv2.detail.ChannelsCompensator`, `cv2.detail.ExposureCompensator`, `cv2.detail.GainCompensator`, `cv2.detail.NoExposureCompensator` | Gain, per-channel and block compensation share an exposure contract with overlap diagnostics. | S1-116–S1-117 |
| `cv2.detail.DpSeamFinder`, `cv2.detail.GraphCutSeamFinder`, `cv::detail::GraphCutSeamFinderBase`, `cv::detail::GraphCutSeamFinderGpu`, `cv2.detail.NoSeamFinder`, `cv2.detail.PairwiseSeamFinder`, `cv2.detail.SeamFinder`, `cv2.detail.VoronoiSeamFinder` | Use selectable seam modes and retain coverage. GPU choices are backend qualification. | S1-49, S1-117 |
| `cv2.detail.Blender`, `cv2.detail.FeatherBlender`, `cv2.detail.MultiBandBlender`, `cv2.detail.createLaplacePyr`, `cv2.detail.createLaplacePyrGpu`, `cv2.detail.createWeightMap`, `cv2.detail.normalizeUsingWeightMap`, `cv2.detail.restoreImageFromLaplacePyr`, `cv2.detail.restoreImageFromLaplacePyrGpu` | Blenders and weight maps share the seam composition contract. `cv.laplacian_pyramid` and `cv.reconstruct_laplacian_pyramid` wrap the CPU detail helpers and floating pyrDown/pyrUp path with signed bands, exact odd sizes and preserved source frames. GPU helper qualification remains open. | S1-117, S1-124 |
| `cv2.detail.Timelapser`, `cv2.detail.TimelapserCrop` | Keep each registered input as a separate frame with common-grid coverage and time provenance. | S1-165 |
| `cv::detail::DisjointSets`, `cv::detail::Graph`, `cv::detail::GraphEdge`, `cv2.detail.overlapRoi`, `cv2.detail.resultRoi`, `cv2.detail.resultRoiIntersection`, `cv2.detail.resultTl`, `cv2.detail.selectRandomSubset`, `cv2.detail.stitchingLogLevel` | Canvas bounds, overlap and internal graph utilities are part of layout, stitching and registered replay. | S1-01, S1-49, S1-164–S1-165 |
| `cv::AffineWarper`, `cv::CompressedRectilinearPortraitWarper`, `cv::CompressedRectilinearWarper`, `cv::CylindricalWarper`, `cv::FisheyeWarper`, `cv::MercatorWarper`, `cv::PaniniPortraitWarper`, `cv::PaniniWarper`, `cv::PlaneWarper`, `cv2.PyRotationWarper`, `cv::SphericalWarper`, `cv::StereographicWarper`, `cv::TransverseMercatorWarper`, `cv2.WarperCreator` | The Python warper dispatches projection choices. C++ factory types follow the same map and coverage contract. | S1-38, S1-104, S1-164 |
| `cv::detail::CompressedRectilinearPortraitProjector`, `cv::detail::CompressedRectilinearProjector`, `cv::detail::CylindricalPortraitProjector`, `cv::detail::CylindricalProjector`, `cv::detail::FisheyeProjector`, `cv::detail::MercatorProjector`, `cv::detail::PaniniPortraitProjector`, `cv::detail::PaniniProjector`, `cv::detail::PlanePortraitProjector`, `cv::detail::PlaneProjector`, `cv2.detail.ProjectorBase`, `cv::detail::SphericalPortraitProjector`, `cv2.detail.SphericalProjector`, `cv::detail::StereographicProjector`, `cv::detail::TransverseMercatorProjector` | Projection implementations remain internal to calibrated panorama and camera estimation. | S1-104, S1-147 |
| `cv::detail::AffineWarper`, `cv::detail::CompressedRectilinearPortraitWarper`, `cv::detail::CompressedRectilinearWarper`, `cv::detail::CylindricalPortraitWarper`, `cv::detail::CylindricalWarper`, `cv::detail::CylindricalWarperGpu`, `cv::detail::FisheyeWarper`, `cv::detail::MercatorWarper`, `cv::detail::PaniniPortraitWarper`, `cv::detail::PaniniWarper`, `cv::detail::PlanePortraitWarper`, `cv::detail::PlaneWarper`, `cv::detail::PlaneWarperGpu`, `cv::detail::RotationWarper`, `cv::detail::RotationWarperBase`, `cv::detail::SphericalPortraitWarper`, `cv::detail::SphericalWarper`, `cv::detail::SphericalWarperGpu`, `cv::detail::StereographicWarper`, `cv::detail::TransverseMercatorWarper` | Map warping and platform variants to reusable layouts. Qualify source coordinates and invalid image support. | S1-38, S1-104, S1-164 |

Sources are the pinned [DNN interface](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/include/opencv2/dnn/dnn.hpp),
[stitching interface and warper factories](https://github.com/opencv/opencv/tree/5.0.0/modules/stitching/include/opencv2),
and [stitching detail headers](https://github.com/opencv/opencv/tree/5.0.0/modules/stitching/include/opencv2/stitching/detail).
The audit includes the public detail declarations and groups platform-specific
variants. Inline implementation files stay under the owning headers.

### Native core and selected contrib API crosswalk

This audit maps 323 named declarations from `core.hpp` and the
public headers reached directly from eight contrib module headers. It spans
62 pinned headers. The Python wheel exposes 289 of the mapped names.
Grouped entries include implementation helpers and unavailable bindings so
their disposition remains explicit. Build qualification is separate from
binding presence. For example, the wheel exposes DynaFu but rejects construction.

#### Core image operations

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.absdiff`, `cv2.add`, `cv2.subtract`, `cv2.multiply`, `cv2.divide`, `cv2.scaleAdd`, `cv2.compare`, `cv2.min`, `cv2.max`, `cv2.inRange` | Typed image arithmetic, difference maps and range decisions own these operations. | S1-15, S1-55 |
| `cv2.addWeighted`, `cv2.bitwise_and`, `cv2.bitwise_not`, `cv2.bitwise_or`, `cv2.bitwise_xor`, `cv2.copyTo` | Masks and blending own the per-pixel composition contract. | S1-16, S1-37, S1-132 |
| `cv2.convertScaleAbs` | Preserve the distinction between signed conversion and absolute display scaling. | S1-13 |
| `cv2.split`, `cv2.merge`, `cv2.mixChannels`, `cv2.extractChannel`, `cv2.insertChannel`, `cv2.transform` | Channel ordering, alpha and image-space mixing follow existing contracts. | S1-14, S1-96 |
| `cv2.LUT`, `cv2.normalize`, `cv2.exp`, `cv2.log`, `cv2.pow`, `cv2.sqrt` | Intensity mapping and output encoding belong to existing image transforms. | S1-17 |
| `cv2.countNonZero`, `cv2.findNonZero`, `cv2.hasNonZero`, `cv2.checkRange`, `cv2.finiteMask`, `cv2.patchNaNs`, `cv2.norm`, `cv2.mean`, `cv2.meanStdDev`, `cv::minMaxIdx`, `cv2.minMaxLoc`, `cv::sum`, `cv2.sumElems` | Image statistics, support, finite-sample checks and replacement share typed masks. | S1-18, S1-20, S1-143 |
| `cv2.reduce`, `cv2.reduceArgMax`, `cv2.reduceArgMin` | Use named axes in image profiles and image-derived score decoding. Keep generic array reductions internal. | S1-20, S2-10, S2-12 |
| `cv2.PSNR` | Pair with the reference-error map and selected support. | S1-55 |
| `cv2.dft`, `cv2.idft`, `cv2.getOptimalDFTSize` | Typed spectra retain mode, padding and original size. S1-135 now qualifies full/phase-only/magnitude-only reconstruction, signed color output, phase floors, normalization and source-window behavior. Zero-phase results receive a new coordinate grid. | S1-41, S1-135 |
| `cv2.dct`, `cv2.idct` | Preserve coefficient scaling and recover the input image. | S1-79 |
| `cv2.mulSpectrums` | Spectrum products stay in filtering, restoration and correlation. | S1-58, S1-80, S1-84 |
| `cv2.divSpectrums` | Verify supported-band gain and phase independently of filter application. | S1-166 |
| `cv2.cartToPolar`, `cv2.polarToCart`, `cv2.phase`, `cv2.magnitude` | Use explicit vector units and angle conventions for gradients and spectral fields. S1-135 uses NumPy hypot/atan2 and conjugate component algebra around native DFT/IDFT, with radians and explicit undefined-phase validity; it does not claim these individual bindings are wrapped. | S1-78, S1-135 |
| `cv2.borderInterpolate`, `cv2.copyMakeBorder` | Border indices are internal to typed padding. | S1-70 |
| `cv2.hconcat`, `cv2.vconcat`, `cv2.repeat` | Use explicit image grids and layout provenance. | S1-37 |
| `cv2.rotate`, `cv2.flip`, `cv2.transpose` | Track the corresponding coordinate transform. | S1-73 |
| `cv2.randn`, `cv2.randu`, `cv2.randShuffle`, `cv2.setRNGSeed`, `cv::theRNG`, `cv::RNG`, `cv::RNG_MT19937` | Graph fixtures must preserve per-run determinism independently of native global RNG state. | S1-105 |
| `cv2.perspectiveTransform` | Point transforms keep their image or workplane frame. | S1-38, S1-101, S3-06 |
| `cv2.kmeans` | Use image clustering and vocabulary contracts. | S1-29, S2-19, S2-44 |
| `cv::PCA`, `cv2.PCACompute`, `cv2.PCACompute2`, `cv2.PCABackProject`, `cv2.PCAProject` | Dimension reduction stays with the image descriptor pipeline. | S2-41 |
| `cv2.Mahalanobis`, `cv2.calcCovarMatrix` | Classical feature scoring needs held-out image fixtures. | S2-43 |
| `cv::LDA`, `cv2.batchDistance` | Treat basic class projections and descriptor distances as implementation options within image models. | S1-44, S2-20–S2-23 |
| `cv2.broadcast`, `cv2.flipND`, `cv2.transposeND`, `cv2.Algorithm`, `cv::Formatted`, `cv::Formatter`, `cv2.SVBackSubst`, `cv::SVD`, `cv2.SVDecomp`, `cv2.completeSymm`, `cv2.determinant`, `cv2.eigen`, `cv2.eigenNonSymmetric`, `cv2.gemm`, `cv2.invert`, `cv2.mulTransposed`, `cv2.setIdentity`, `cv2.solve`, `cv2.solveCubic`, `cv2.solvePoly`, `cv2.sort`, `cv2.sortIdx`, `cv::swap`, `cv2.trace` | Keep generic array layout, scalar algebra and solver internals inside the image operations that use them. | — |

#### Contrib filters, texture and feature tools

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.ximgproc.anisotropicDiffusion`, `cv2.ximgproc.niBlackThreshold`, `cv2.ximgproc.thinning` | Preserve threshold formula, skeleton and diffusion-specific image checks. | S4-02, S4-06, S4-39 |
| `cv::ximgproc::BrightEdges` | Compare local-contrast edges with known reference geometry using a qualified build or fallback. | S4-86 |
| `cv2.ximgproc.colorMatchTemplate`, `cv2.ximgproc.createQuaternionImage`, `cv2.ximgproc.qconj`, `cv2.ximgproc.qdft`, `cv2.ximgproc.qmultiply`, `cv2.ximgproc.qunitary` | Quaternion helpers stay inside the color-template workflow. | S4-85 |
| `cv2.ximgproc.GradientDericheX`, `cv2.ximgproc.GradientDericheY`, `cv::ximgproc::GradientPaillouX`, `cv::ximgproc::GradientPaillouY` | Both gradient families use the signed-axis and localization fixture. | S4-41 |
| `cv2.ximgproc.EdgeDrawing`, `cv2.ximgproc.EdgeDrawing.Params`, `cv2.ximgproc.createEdgeDrawing` | Group parameter records with edge and primitive detection. | S4-79 |
| `cv2.ximgproc.AdaptiveManifoldFilter`, `cv2.ximgproc.amFilter`, `cv2.ximgproc.createAMFilter` | Qualify edge-preserving manifold smoothing on selected images. | S4-37 |
| `cv2.ximgproc.DTFilter`, `cv2.ximgproc.dtFilter`, `cv2.ximgproc.createDTFilter` | Compare domain-transform modes on the same image and guide. | S4-27 |
| `cv2.ximgproc.FastBilateralSolverFilter`, `cv2.ximgproc.createFastBilateralSolverFilter`, `cv2.ximgproc.fastBilateralSolverFilter` | Verify confidence-weighted smoothing and supported input build. | S4-36 |
| `cv2.ximgproc.FastGlobalSmootherFilter`, `cv2.ximgproc.createFastGlobalSmootherFilter`, `cv2.ximgproc.fastGlobalSmootherFilter` | Separate guide handling and edge retention from alternate implementations. | S4-13 |
| `cv2.ximgproc.GuidedFilter`, `cv2.ximgproc.createGuidedFilter`, `cv2.ximgproc.guidedFilter` | Qualify filter reuse and the single-call variant together. | S4-01 |
| `cv2.ximgproc.jointBilateralFilter`, `cv2.ximgproc.rollingGuidanceFilter`, `cv2.ximgproc.l0Smooth`, `cv2.ximgproc.bilateralTextureFilter` | Keep distinct acceptance for guide-dependent, iterative and texture-specific smoothing. | S4-14–S4-15, S4-40, S4-88 |
| `cv2.ximgproc.EdgeBoxes`, `cv2.ximgproc.createEdgeBoxes`, `cv2.ximgproc.RFFeatureGetter`, `cv2.ximgproc.StructuredEdgeDetection`, `cv2.ximgproc.createRFFeatureGetter`, `cv2.ximgproc.createStructuredEdgeDetection` | Use model edges and geometry for reproducible proposals. | S4-44–S4-45 |
| `cv2.ximgproc.edgePreservingFilter`, `cv2.ximgproc.weightedMedianFilter` | Qualify separate weight or variance behavior on inspection edges. | S4-38, S4-78 |
| `cv2.ximgproc.covarianceEstimation` | Qualify spatial patch covariance with bounded memory and a direct reference. | S4-90 |
| `cv2.ximgproc.FastHoughTransform`, `cv2.ximgproc.HoughPoint2Line`, `cv2.ximgproc.RadonTransform` | Separate line-geometry and angular projection measurements. | S4-43, S4-87 |
| `cv2.ximgproc.FastLineDetector`, `cv2.ximgproc.createFastLineDetector`, `cv2.ximgproc.findEllipses`, `cv2.ximgproc.RidgeDetectionFilter` | Preserve detector outputs and source geometry. | S4-05, S4-18, S4-42 |
| `cv2.ximgproc.ContourFitting`, `cv2.ximgproc.contourSampling`, `cv2.ximgproc.createContourFitting`, `cv2.ximgproc.fourierDescriptor`, `cv2.ximgproc.transformFD`, `cv::ximgproc::rl::createRLEImage`, `cv::ximgproc::rl::dilate`, `cv::ximgproc::rl::erode`, `cv::ximgproc::rl::getStructuringElement`, `cv::ximgproc::rl::isRLMorphologyPossible`, `cv::ximgproc::rl::morphologyEx`, `cv::ximgproc::rl::paint`, `cv::ximgproc::rl::threshold` | Use spectral contour and sparse-mask contracts with explicit fixture parity. | S4-82–S4-83 |
| `cv2.ximgproc.SuperpixelLSC`, `cv2.ximgproc.createSuperpixelLSC`, `cv2.ximgproc.ScanSegment`, `cv2.ximgproc.createScanSegment`, `cv2.ximgproc.SuperpixelSEEDS`, `cv2.ximgproc.createSuperpixelSEEDS`, `cv2.ximgproc.SuperpixelSLIC`, `cv2.ximgproc.createSuperpixelSLIC` | Compare native labels and boundary masks on a common segmentation fixture. | S4-03, S4-28–S4-30, S4-89 |
| `cv2.ximgproc.segmentation.GraphSegmentation`, `cv2.ximgproc.segmentation.SelectiveSearchSegmentation`, `cv2.ximgproc.segmentation.SelectiveSearchSegmentationStrategy`, `cv2.ximgproc.segmentation.SelectiveSearchSegmentationStrategyColor`, `cv2.ximgproc.segmentation.SelectiveSearchSegmentationStrategyFill`, `cv2.ximgproc.segmentation.SelectiveSearchSegmentationStrategyMultiple`, `cv2.ximgproc.segmentation.SelectiveSearchSegmentationStrategySize`, `cv2.ximgproc.segmentation.SelectiveSearchSegmentationStrategyTexture`, `cv2.ximgproc.segmentation.createGraphSegmentation`, `cv2.ximgproc.segmentation.createSelectiveSearchSegmentation`, `cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor`, `cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill`, `cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple`, `cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize`, `cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture` | Group graph labels and selective-search strategies under their respective outputs. | S4-04, S4-31 |
| `cv2.ximgproc.PeiLinNormalization`, `cv2.ximgproc.EdgeAwareInterpolator`, `cv2.ximgproc.RICInterpolator`, `cv2.ximgproc.SparseMatchInterpolator`, `cv2.ximgproc.createEdgeAwareInterpolator`, `cv2.ximgproc.createRICInterpolator` | Keep independent color normalization and motion interpolation checks. | S4-59, S4-69, S4-77 |
| `cv2.xfeatures2d.FREAK`, `cv2.xfeatures2d.BriefDescriptorExtractor`, `cv2.xfeatures2d.DAISY`, `cv2.xfeatures2d.LATCH`, `cv2.xfeatures2d.BEBLID`, `cv2.xfeatures2d.TEBLID`, `cv2.xfeatures2d.BoostDesc`, `cv2.xfeatures2d.LUCID`, `cv2.xfeatures2d.VGG` | Use keypoint-aware descriptor records with matching fixtures for binary and float features. | S4-19–S4-20, S4-48, S4-74–S4-75 |
| `cv2.xfeatures2d.HarrisLaplaceFeatureDetector`, `cv2.xfeatures2d.MSDDetector`, `cv2.xfeatures2d.StarDetector`, `cv2.xfeatures2d.TBMR`, `cv2.xfeatures2d.AffineFeature2D`, `cv::xfeatures2d::Elliptic_KeyPoint` | Map regional or scale-aware feature variants to an existing detection contract. | S4-49, S4-80, S1-128 |
| `cv::xfeatures2d::AGAST`, `cv2.xfeatures2d.AgastFeatureDetector`, `cv::xfeatures2d::FASTForPointSet` | Share the feature detection and selected-point workflow. | S1-42 |
| `cv2.xfeatures2d.AKAZE`, `cv2.xfeatures2d.KAZE`, `cv2.xfeatures2d.BRISK` | These descriptor classes are under xfeatures2d on the pinned wheel. | S1-43 |
| `cv2.xfeatures2d.PCTSignatures`, `cv2.xfeatures2d.PCTSignaturesSQFD` | Qualify reusable patch signatures and distribution-distance scoring together. | S4-91 |
| `cv2.xfeatures2d.matchGMS`, `cv2.xfeatures2d.matchLOGOS`, `cv2.xfeatures2d.BOWTrainer`, `cv2.xfeatures2d.BOWKMeansTrainer`, `cv2.xfeatures2d.BOWImgDescriptorExtractor` | Map vocabulary training and distinct geometric verification modes. | S2-44, S4-47, S4-92 |
| `cv2.xfeatures2d.SURF` | Defer restricted SURF execution until an explicitly qualified build and inspection fixture justify it. | — |

#### Contrib motion, restoration and image quality

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.optflow.DualTVL1OpticalFlow`, `cv2.optflow.calcOpticalFlowSF`, `cv2.optflow.calcOpticalFlowSparseToDense`, `cv2.optflow.createOptFlow_DeepFlow`, `cv2.optflow.createOptFlow_DualTVL1`, `cv2.optflow.createOptFlow_Farneback`, `cv2.optflow.createOptFlow_SimpleFlow`, `cv2.optflow.createOptFlow_SparseToDense` | Group factory and direct-call variants under the corresponding motion workflow. | S1-51, S4-22–S4-23, S4-57–S4-58 |
| `cv2.optflow.DenseRLOFOpticalFlow`, `cv2.optflow.RLOFOpticalFlowParameter`, `cv2.optflow.SparseRLOFOpticalFlow`, `cv2.optflow.calcOpticalFlowDenseRLOF`, `cv2.optflow.calcOpticalFlowSparseRLOF`, `cv2.optflow.createOptFlow_DenseRLOF`, `cv2.optflow.createOptFlow_SparseRLOF` | Preserve sparse versus dense output contracts and motion-boundary checks. | S4-24, S4-84 |
| `cv2.optflow.OpticalFlowPCAFlow`, `cv2.optflow.PCAPrior`, `cv2.optflow.createOptFlow_PCAFlow`, `cv2.motempl.calcGlobalOrientation`, `cv2.motempl.calcMotionGradient`, `cv2.motempl.segmentMotion`, `cv2.motempl.updateMotionHistory` | Keep trained prior options internal to PCAFlow and time-surface operations together. | S4-55, S4-58 |
| `cv2.optflow.GPCDetails`, `cv2.optflow.GPCPatchDescriptor`, `cv2.optflow.GPCPatchSample`, `cv2.optflow.GPCTrainingSamples`, `cv2.optflow.GPCTree`, `cv::read`, `cv::write` | Defer GPC training and persistence helpers until a usable end-to-end matcher and pinned fixture are available. | — |
| `cv2.xphoto.bm3dDenoising`, `cv2.xphoto.dctDenoising`, `cv2.xphoto.inpaint` | Separate restoration methods with damage, noise and mask fixtures. | S4-16–S4-17, S4-62 |
| `cv2.xphoto.oilPainting`, `cv2.xphoto.TonemapDurand`, `cv2.xphoto.createTonemapDurand`, `cv2.xphoto.GrayworldWB`, `cv2.xphoto.LearningBasedWB`, `cv2.xphoto.SimpleWB`, `cv2.xphoto.WhiteBalancer`, `cv2.xphoto.applyChannelGains`, `cv2.xphoto.createGrayworldWB`, `cv2.xphoto.createLearningBasedWB`, `cv2.xphoto.createSimpleWB` | Group classes, factories and channel gains with the relevant photo workflow. | S4-12, S4-63–S4-66 |
| `cv2.quality.QualityBase`, `cv2.quality.QualityMSE`, `cv2.quality.QualityPSNR`, `cv2.quality.QualitySSIM`, `cv2.quality.QualityGMSD`, `cv2.quality.QualityBRISQUE` | Share a quality contract with declared support, maps where available, and model assets for no-reference scoring. | S1-55, S4-07–S4-08, S4-32 |

#### Contrib depth and surface reconstruction

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.ximgproc.DisparityFilter`, `cv2.ximgproc.DisparityWLSFilter`, `cv2.ximgproc.createDisparityWLSFilter`, `cv2.ximgproc.createDisparityWLSFilterGeneric`, `cv2.ximgproc.createRightMatcher` | Split built-in matcher filtering from filtering imported disparity with explicit scale and confidence. | S3-16, S3-121 |
| `cv2.ximgproc.computeBadPixelPercent`, `cv2.ximgproc.computeMSE`, `cv2.ximgproc.getDisparityVis`, `cv2.ximgproc.readGT` | Ground-truth decoding, fixed-point metrics and review graphics belong to stereo evaluation. | S3-48 |
| `cv::stereo::StereoBinaryBM`, `cv::stereo::StereoBinarySGBM`, `cv::stereo::censusTransform`, `cv::stereo::modifiedCensusTransform`, `cv::stereo::starCensusTransform`, `cv::stereo::symetricCensusTransform` | Binary matching and its cost diagnostics need qualified C++ access or a tested equivalent. | S3-61, S3-122 |
| `cv2.stereo.MatchQuasiDense`, `cv2.stereo.PropagationParameters`, `cv2.stereo.QuasiDenseStereo` | Keep propagation, disparity and correspondence fields in the same calibrated pair. | S3-72 |
| `cv2.structured_light.GrayCodePattern`, `cv::structured_light::GrayCodePattern::Params`, `cv2.structured_light.SinusoidalPattern`, `cv2.structured_light.SinusoidalPattern.Params`, `cv2.structured_light.StructuredLightPattern` | Group pattern and parameter types with decoding, confidence masks and projector correspondence. | S3-28–S3-30, S3-67 |
| `cv2.colored_kinfu.ColoredKinFu`, `cv2.colored_kinfu.Params`, `cv2.kinfu.KinFu`, `cv2.kinfu.Params`, `cv2.kinfu.VolumeParams`, `cv2.large_kinfu.LargeKinfu`, `cv2.large_kinfu.Params`, `cv2.large_kinfu.VolumeParams` | Rigid tracking, color and volume size follow their respective fusion workflows. | S3-26–S3-27, S3-71, S3-93–S3-95 |
| `cv2.dynafu.DynaFu` | The class is present but constructor execution is disabled in the pinned wheel. Use a qualified backend or equivalent before claiming support. | S3-123 |
| `cv2.linemod.ColorGradient`, `cv2.linemod.DepthNormal`, `cv2.linemod.Detector`, `cv2.linemod.Feature`, `cv2.linemod.Match`, `cv2.linemod.Modality`, `cv2.linemod.QuantizedPyramid`, `cv2.linemod.Template`, `cv2.linemod.colormap`, `cv2.linemod.drawFeatures`, `cv2.linemod.getDefaultLINE`, `cv2.linemod.getDefaultLINEMOD` | Modalities, templates, match records and graphics stay with the trained RGB-D pose contract. | S3-69 |

[Core source header](https://github.com/opencv/opencv/blob/5.0.0/modules/core/include/opencv2/core.hpp),
[extended image processing](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/ximgproc/include/opencv2),
[extended features](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/xfeatures2d/include/opencv2),
[photo](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/xphoto/include/opencv2), [flow](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/optflow/include/opencv2),
[quality](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/quality/include/opencv2), [stereo](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/xstereo/include/opencv2),
[structured light](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/structured_light/include/opencv2), and
[RGB-D reconstruction](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/rgbd/include/opencv2).

### Native reconstruction, registration and stabilization API crosswalk

This audit maps 115 named declarations from 43 pinned headers across five
contrib families. The wheel exposes 23 names, concentrated in omnidirectional
calibration, phase unwrapping and image registration. The SfM, custom-target
and video stabilization classes require a different backend or a tested equivalent.

#### Calibration, reconstruction and phase

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv::ccalib::CustomPattern`, `cv::randpattern::RandomPatternCornerFinder`, `cv::randpattern::RandomPatternGenerator`, `cv::multicalib::MultiCameraCalibration` | Separate repeatable target preparation and partial-view detection from the rig fit. These source classes lack direct wheel bindings. | S3-70, S3-126 |
| `cv2.omnidir.calibrate` | Fit a lens model with held-out edge reprojections. | S3-05 |
| `cv2.omnidir.initUndistortRectifyMap`, `cv2.omnidir.projectPoints`, `cv2.omnidir.undistortImage`, `cv2.omnidir.undistortPoints` | Share the lens projection contract with camera rays, calibrated remap and coverage. | S3-05, S3-09, S3-55 |
| `cv2.omnidir.stereoCalibrate`, `cv2.omnidir.stereoRectify` | Fit and rectify the wide-angle pair before disparity or depth validation. | S3-14 |
| `cv2.omnidir.stereoReconstruct` | Qualify wide-angle disparity and metric cloud output together. | S3-124 |
| `cv2.phase_unwrapping.HistogramPhaseUnwrapping`, `cv2.phase_unwrapping.HistogramPhaseUnwrapping.Params`, `cv2.phase_unwrapping.PhaseUnwrapping` | Keep phase output, shadow support and inverse reliability in a calibrated fringe workflow. | S3-67 |
| `cv::sfm::applyTransformationToPoints`, `cv::sfm::isotropicPreconditionerFromPoints`, `cv::sfm::normalizeIsotropicPoints`, `cv::sfm::normalizePoints`, `cv::sfm::preconditionerFromPoints`, `cv::sfm::meanAndVarianceAlongRows`, `cv::sfm::skew` | Conditioning and solver numerics stay inside the fitted reconstruction. | S3-33 |
| `cv::sfm::computeOrientation`, `cv::sfm::essentialFromFundamental`, `cv::sfm::essentialFromRt`, `cv::sfm::fundamentalFromEssential`, `cv::sfm::fundamentalFromProjections`, `cv::sfm::motionFromEssential`, `cv::sfm::motionFromEssentialChooseSolution`, `cv::sfm::normalizeFundamental`, `cv::sfm::normalizedEightPointSolver`, `cv::sfm::projectionsFromFundamental`, `cv::sfm::relativeCameraMotion`, `cv::sfm::fundamentalFromCorrespondences7PointRobust`, `cv::sfm::fundamentalFromCorrespondences8PointRobust` | Group fundamental fitting, outlier rejection and pose candidate selection by calibrated image correspondence. | S3-11, S3-33, S3-43 |
| `cv::sfm::KRtFromProjection`, `cv::sfm::depth`, `cv::sfm::euclideanToHomogeneous`, `cv::sfm::homogeneousToEuclidean`, `cv::sfm::projectionFromKRt`, `cv::sfm::triangulatePoints` | Apply projection and metric triangulation with frame-aware outputs. The shipped triangulator takes two views; more views need a qualified solver because native sfm bindings are absent. | S3-09, S3-12, S3-33, S3-65, S3-117 |
| `cv::sfm::reconstruct`, `cv::sfm::BaseSFM`, `cv::sfm::SFMLibmvEuclideanReconstruction`, `cv::sfm::libmv_CameraIntrinsicsOptions`, `cv::sfm::libmv_ReconstructionOptions` | Qualify a reconstruction backend with a measured camera sequence. The pinned Python wheel lacks these native SfM bindings. | S3-33, S3-65 |
| `cv::sfm::importReconstruction` | Bundler is the native documented import format. Verify camera and sparse point conventions together. | S3-125 |

#### Registration and video stabilization

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.reg.Map`, `cv2.reg.MapAffine`, `cv2.reg.MapProjec`, `cv2.reg.MapShift`, `cv2.reg.Mapper`, `cv2.reg.MapperGradAffine`, `cv2.reg.MapperGradEuclid`, `cv2.reg.MapperGradProj`, `cv2.reg.MapperGradShift`, `cv2.reg.MapperGradSimilar`, `cv2.reg.MapTypeCaster`, `cv2.reg.MapperPyramid` | Share map application, inverse and coverage across registration variants. Retain each model's transform constraints. | S4-72 |
| `cv::videostab::FromFileMotionReader`, `cv::videostab::ImageMotionEstimatorBase`, `cv::videostab::KeypointBasedMotionEstimator`, `cv::videostab::KeypointBasedMotionEstimatorGpu`, `cv::videostab::MotionEstimatorBase`, `cv::videostab::MotionEstimatorL1`, `cv::videostab::MotionEstimatorRansacL2`, `cv::videostab::ToFileMotionWriter`, `cv::videostab::estimateGlobalMotionLeastSquares`, `cv::videostab::estimateGlobalMotionRansac`, `cv::videostab::getMotion` | Robust motion estimation and reusable trajectories have separate fixtures. Track source frame indices and transform direction. | S1-02, S1-50, S1-167 |
| `cv::videostab::OnePassStabilizer`, `cv::videostab::StabilizerBase`, `cv::videostab::TwoPassStabilizer`, `cv::videostab::GaussianMotionFilter`, `cv::videostab::IMotionStabilizer`, `cv::videostab::LpMotionStabilizer`, `cv::videostab::MotionFilterBase`, `cv::videostab::MotionStabilizationPipeline`, `cv::videostab::ensureInclusionConstraint`, `cv::videostab::estimateOptimalTrimRatio`, `cv::videostab::RansacParams`, `cv::videostab::IOutlierRejector`, `cv::videostab::NullOutlierRejector`, `cv::videostab::TranslationBasedLocalOutlierRejector` | Group path smoothing, correction bounds and outlier rejection in one- and two-pass stabilization. Keep memory, crop and coverage testable. | S1-02, S1-50 |
| `cv::videostab::DensePyrLkOptFlowEstimatorGpu`, `cv::videostab::IDenseOptFlowEstimator`, `cv::videostab::ISparseOptFlowEstimator`, `cv::videostab::PyrLkOptFlowEstimatorBase`, `cv::videostab::SparsePyrLkOptFlowEstimator`, `cv::videostab::SparsePyrLkOptFlowEstimatorGpu` | Keep native estimator variants behind the existing optical-flow and pyramid workflows. | S1-51, S1-155 |
| `cv::videostab::ColorAverageInpainter`, `cv::videostab::ColorInpainter`, `cv::videostab::ConsistentMosaicInpainter`, `cv::videostab::InpainterBase`, `cv::videostab::InpaintingPipeline`, `cv::videostab::MotionInpainter`, `cv::videostab::NullInpainter`, `cv::videostab::calcFlowMask`, `cv::videostab::completeFrameAccordingToFlow`, `cv::videostab::FastMarchingMethod` | Keep filled review pixels distinct from measured support. | S1-148 |
| `cv::videostab::DeblurerBase`, `cv::videostab::NullDeblurer`, `cv::videostab::WeightingDeblurer`, `cv::videostab::calcBlurriness` | Recover detail from aligned neighboring frames with source support and motion checks. | S1-168 |
| `cv::videostab::MoreAccurateMotionWobbleSuppressor`, `cv::videostab::MoreAccurateMotionWobbleSuppressorBase`, `cv::videostab::MoreAccurateMotionWobbleSuppressorGpu`, `cv::videostab::NullWobbleSuppressor`, `cv::videostab::WobbleSuppressorBase` | Qualify correction of row-dependent motion separately from global path smoothing. | S1-169 |
| `cv::videostab::IFrameSource`, `cv::videostab::NullFrameSource`, `cv::videostab::VideoFileSource`, `cv::videostab::ILog`, `cv::videostab::LogToStdout`, `cv::videostab::NullLog` | Frame sources and logger utilities stay under application acquisition and diagnostics. | — |

Pinned sources are [camera calibration](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/ccalib/include/opencv2),
[structure from motion](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/sfm/include/opencv2),
[phase unwrapping](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/phase_unwrapping/include/opencv2),
[registration](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/reg/include/opencv2), and
[video stabilization](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/videostab/include/opencv2).

### Native OCR, shape, hashing, foreground, saliency and super-resolution crosswalk

This audit maps 99 named declarations from 26 pinned public headers.
The installed Python wheel exposes 77. Constructors, settings and instance
methods follow their owning workflow. Binding presence is separate from
a working backend and reproducible graph result.

#### Scene text

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.text.ERFilter`, `cv2.text.ERFilter.Callback`, `cv::text::ERStat`, `cv::text::MSERsToERStats`, `cv2.text.computeNMChannels`, `cv2.text.createERFilterNM1`, `cv2.text.createERFilterNM2`, `cv2.text.detectRegions`, `cv2.text.erGrouping`, `cv2.text.loadClassifierNM1`, `cv2.text.loadClassifierNM2` | Keep extremal-region extraction, classifier loading and grouping in the same text-region contract. | S4-67 |
| `cv2.text.detectTextSWT` | Qualify stroke-width detection and its grouped boxes without trained detector assets. | S4-93 |
| `cv2.text.TextDetector`, `cv2.text.TextDetectorCNN` | Use an image-to-text-region model contract and a pinned detection fixture. | S2-32 |
| `cv2.text.BaseOCR`, `cv2.text.OCRTesseract` | Validate the linked backend and language data. The audited wheel reports Tesseract unavailable and returns empty text on a known label. | S4-68 |
| `cv2.text.OCRHMMDecoder`, `cv2.text.OCRHMMDecoder.ClassifierCallback`, `cv2.text.OCRBeamSearchDecoder`, `cv2.text.OCRBeamSearchDecoder.ClassifierCallback`, `cv2.text.loadOCRHMMClassifier`, `cv2.text.loadOCRHMMClassifierNM`, `cv2.text.loadOCRHMMClassifierCNN`, `cv2.text.loadOCRBeamSearchClassifierCNN`, `cv2.text.createOCRHMMTransitionsTable` | Group native decoder helpers with their classifiers and calibrated output records. | S4-94 |
| `cv::text::OCRHolisticWordRecognizer` | Defer this C++ recognizer pending a usable model and a distinct inspection requirement. | — |

#### Contours and image hashing

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.ShapeDistanceExtractor`, `cv2.ShapeContextDistanceExtractor`, `cv2.createShapeContextDistanceExtractor` | Keep geometry-only scoring separate from appearance-assisted ranking with paired source images. | S4-50, S4-95 |
| `cv2.HausdorffDistanceExtractor`, `cv2.createHausdorffDistanceExtractor` | Check the rank and metric settings on controlled contours. | S4-51 |
| `cv2.ChiHistogramCostExtractor`, `cv2.EMDHistogramCostExtractor`, `cv2.EMDL1HistogramCostExtractor`, `cv2.HistogramCostExtractor`, `cv2.NormHistogramCostExtractor`, `cv2.createChiHistogramCostExtractor`, `cv2.createEMDHistogramCostExtractor`, `cv2.createEMDL1HistogramCostExtractor`, `cv2.createNormHistogramCostExtractor`, `cv::EMDL1` | Use histogram costs inside shape scoring and distribution comparison rather than exposing matrix utilities. | S1-120, S4-50 |
| `cv2.AffineTransformer`, `cv2.ShapeTransformer`, `cv2.ThinPlateSplineShapeTransformer`, `cv2.createAffineTransformer`, `cv2.createThinPlateSplineShapeTransformer` | Affine choices belong to reference alignment and shape scoring. Thin-plate spline warping owns the nonrigid fixture. | S1-48, S4-50, S4-52, S4-95 |
| `cv2.img_hash.ImgHashBase`, `cv2.img_hash.AverageHash`, `cv2.img_hash.averageHash`, `cv2.img_hash.BlockMeanHash`, `cv2.img_hash.blockMeanHash`, `cv2.img_hash.MarrHildrethHash`, `cv2.img_hash.marrHildrethHash`, `cv2.img_hash.PHash`, `cv2.img_hash.pHash` | Pairwise binary hash comparison is shipped. Catalog persistence and ranked retrieval need their own checks. | S4-09, S4-96 |
| `cv2.img_hash.ColorMomentHash`, `cv2.img_hash.colorMomentHash`, `cv2.img_hash.RadialVarianceHash`, `cv2.img_hash.radialVarianceHash` | Preserve the native descriptor encoding and comparison semantics through catalog storage and retrieval. | S4-10, S4-96 |

#### Foreground and saliency

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.bgsegm.BackgroundSubtractorGSOC`, `cv2.bgsegm.createBackgroundSubtractorGSOC` | Group model creation with warmup and foreground outputs. | S4-21 |
| `cv2.bgsegm.BackgroundSubtractorLSBP`, `cv2.bgsegm.BackgroundSubtractorLSBPDesc`, `cv2.bgsegm.createBackgroundSubtractorLSBP` | Keep local descriptors internal to the LSBP foreground path. | S4-33 |
| `cv2.bgsegm.BackgroundSubtractorCNT`, `cv2.bgsegm.BackgroundSubtractorGMG`, `cv2.bgsegm.BackgroundSubtractorMOG`, `cv2.bgsegm.createBackgroundSubtractorCNT`, `cv2.bgsegm.createBackgroundSubtractorGMG`, `cv2.bgsegm.createBackgroundSubtractorMOG` | Qualify alternative adaptive models on shared replay. Treat warm-start images and replay as graph data. | S1-170, S4-56 |
| `cv2.bgsegm.SyntheticSequenceGenerator`, `cv2.bgsegm.createSyntheticSequenceGenerator` | Use seeded clips with reference masks for replay fixtures. | S4-76 |
| `cv2.saliency.Saliency`, `cv2.saliency.StaticSaliency`, `cv2.saliency.StaticSaliencyFineGrained`, `cv2.saliency.StaticSaliencySpectralResidual` | Use typed saliency maps and a repeatable mask conversion. | S4-53 |
| `cv2.saliency.MotionSaliency`, `cv2.saliency.MotionSaliencyBinWangApr2014` | Verify image size, warmup and reset on timestamped frames. | S4-54 |
| `cv2.saliency.Objectness`, `cv2.saliency.ObjectnessBING` | Qualify ranked proposals and their training assets separately from saliency maps. | S4-97 |

#### Super-resolution

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.dnn_superres.DnnSuperResImpl` | Use model-backed single-scale and multi-scale contracts. Multi-output symbol presence leaves Python result delivery and model execution to qualification. | S2-35, S2-58 |
| `cv::superres::FrameSource`, `cv::superres::SuperResolution`, `cv::superres::createFrameSource_Camera`, `cv::superres::createFrameSource_Empty`, `cv::superres::createFrameSource_Video`, `cv::superres::createFrameSource_Video_CUDA`, `cv::superres::createSuperResolution_BTVL1`, `cv::superres::createSuperResolution_BTVL1_CUDA` | Frame sources and accelerator paths stay within sequence reconstruction. The pinned wheel exposes no direct Python bindings for these entry points. | S4-81 |
| `cv::superres::BroxOpticalFlow`, `cv::superres::DenseOpticalFlowExt`, `cv::superres::DualTVL1OpticalFlow`, `cv::superres::FarnebackOpticalFlow`, `cv::superres::createOptFlow_Brox_CUDA`, `cv::superres::createOptFlow_DualTVL1`, `cv::superres::createOptFlow_DualTVL1_CUDA`, `cv::superres::createOptFlow_Farneback`, `cv::superres::createOptFlow_Farneback_CUDA`, `cv::superres::createOptFlow_PyrLK_CUDA` | Keep native motion estimation choices behind aligned-frame reconstruction and qualified backend variants. | S1-51, S4-81 |

Pinned sources are [scene text](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/text/include/opencv2),
[shape matching](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/shape/include/opencv2),
[image hashing](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/img_hash/include/opencv2),
[background segmentation](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/bgsegm/include/opencv2),
[saliency](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/saliency/include/opencv2), and
[classical](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/superres/include/opencv2) and
[learned](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/dnn_superres/include/opencv2) super-resolution.

### Native image ML, tracking, detection and restoration crosswalk

This audit maps 123 named declarations from 33 pinned public headers in
13 contrib modules. The installed Python wheel exposes 77 mapped names.
Nested settings, instance methods and implementation-only utilities follow
their owning graph workflows. Generic tabular learning remains outside scope.

#### Image-derived classical learning

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.ml.SVM`, `cv2.ml.SVMSGD` | Keep classification, regression, one-class and linear detector training distinct by their image contracts. SGD is an optional linear trainer. | S2-20, S2-49, S2-60–S2-61 |
| `cv2.ml.KNearest`, `cv2.ml.EM` | Verify held-out image or pixel features and persisted model parameters. | S2-21, S2-42 |
| `cv2.ml.DTrees`, `cv2.ml.RTrees`, `cv2.ml.Boost` | Group tree-based classification, score checks and model variants by the same image validation contract. | S2-22–S2-23 |
| `cv2.ml.NormalBayesClassifier`, `cv2.ml.LogisticRegression`, `cv2.ml.ANN_MLP` | Use typed image descriptors and qualify score semantics and calibration. Native Bayes scores on a synthetic fixture did not sum to one. | S2-06, S2-23, S2-59 |
| `cv2.ml.TrainData`, `cv2.ml.StatModel`, `cv2.ml.ParamGrid` | Keep model training, selection, reload and held-out splits under the image model workflow. | S2-20–S2-23, S2-59–S2-61 |
| `cv::ml::createConcentricSpheresTestSet`, `cv::ml::randMVNormal` | Generic numerical test-data generators stay outside the inspection graph checklist. | — |

#### Tracking and detection

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.TrackerCSRT`, `cv2.TrackerKCF` | CSRT and KCF box tracking are shipped. CSRT already consumes an optional initial mask. | S1-53 |
| `cv::detail::tracking::kalman_filters::AugmentedUnscentedKalmanFilterParams`, `cv::detail::tracking::kalman_filters::UkfSystemModel`, `cv::detail::tracking::kalman_filters::UnscentedKalmanFilter`, `cv::detail::tracking::kalman_filters::UnscentedKalmanFilterParams`, `cv::detail::tracking::kalman_filters::createAugmentedUnscentedKalmanFilter`, `cv::detail::tracking::kalman_filters::createUnscentedKalmanFilter` | Keep linear track prediction separate from qualified nonlinear orientation and scale prediction. These native contrib declarations have no direct wheel bindings. | S1-54, S1-171 |
| `cv2.legacy.MultiTracker`, `cv::legacy::MultiTrackerTLD`, `cv::legacy::MultiTracker_Alt`, `cv2.legacy.Tracker`, `cv2.legacy.TrackerBoosting`, `cv2.legacy.TrackerCSRT`, `cv2.legacy.TrackerKCF`, `cv2.legacy.TrackerMIL`, `cv2.legacy.TrackerMOSSE`, `cv2.legacy.TrackerMedianFlow`, `cv2.legacy.TrackerTLD`, `cv2.legacy.upgradeTrackingAPI` | Keep legacy tracker choices and multi-object containers behind seeded tracking or detection association. Require a distinct fixture before promoting a separate variant. | S1-53, S1-156, S2-28 |
| `cv::detail::tracking::tbm::CosDistance`, `cv::detail::tracking::tbm::IDescriptorDistance`, `cv::detail::tracking::tbm::IImageDescriptor`, `cv::detail::tracking::tbm::ITrackerByMatching`, `cv::detail::tracking::tbm::MatchTemplateDistance`, `cv::detail::tracking::tbm::ResizedImageDescriptor`, `cv::detail::tracking::tbm::Track`, `cv::detail::tracking::tbm::TrackedObject`, `cv::detail::tracking::tbm::TrackerParams`, `cv::detail::tracking::tbm::createTrackerByMatching` | Descriptor association and track history follow the multi-object identity contract. The native interfaces lack direct bindings in the pinned wheel. | S2-28 |
| `cv::detail::tracking::TrackerContribFeature`, `cv::detail::tracking::TrackerContribFeatureHAAR`, `cv::detail::tracking::TrackerContribFeatureSet`, `cv::detail::tracking::TrackerContribSampler`, `cv::detail::tracking::TrackerContribSamplerAlgorithm`, `cv::detail::tracking::TrackerContribSamplerCSC`, `cv::detail::tracking::TrackerFeatureFeature2d`, `cv::detail::tracking::TrackerFeatureHOG`, `cv::detail::tracking::TrackerFeatureLBP`, `cv::detail::tracking::TrackerSamplerCS`, `cv::detail::tracking::TrackerSamplerPF`, `cv::detail::tracking::TrackerStateEstimatorAdaBoosting`, `cv::detail::tracking::TrackerStateEstimatorSVM`, `cv::detail::tracking::tld::tld_InitDataset`, `cv::detail::tracking::tld::tld_getNextDatasetFrame` | Tracker samplers, classifiers and legacy dataset glue stay internal. | — |
| `cv::detail::tracking::computeInteractionMatrix`, `cv::detail::tracking::computeTwist`, `cv2.rapid.GOSTracker`, `cv2.rapid.OLSTracker`, `cv2.rapid.Rapid`, `cv2.rapid.Tracker`, `cv2.rapid.convertCorrespondencies`, `cv2.rapid.drawCorrespondencies`, `cv2.rapid.drawSearchLines`, `cv2.rapid.drawWireframe`, `cv2.rapid.extractControlPoints`, `cv2.rapid.extractLineBundle`, `cv2.rapid.findCorrespondencies`, `cv2.rapid.rapid` | Keep pose iteration, line correspondences and drawing in the metric silhouette tracking workflow. | S3-68 |
| `cv::DetectionBasedTracker` | Periodic full-frame search and local refinement have their own detector budget and recovery fixture. No direct Python binding is exposed. | S1-172 |
| `cv2.BaseCascadeClassifier`, `cv2.CascadeClassifier` | Expose inspection cascades through the existing model input and detection contract. | S2-47 |
| `cv2.HOGDescriptor` | Separate extracting descriptors, training a detector and running multi-scale inference. | S2-46, S2-61, S4-26 |
| `cv2.groupRectangles`, `cv::groupRectangles_meanshift`, `cv::SimilarRects` | Preserve vote counts and output geometry. Alternative clustering remains a backend choice. | S2-62 |
| `cv::createFaceDetectionMaskGenerator`, `cv::xobjdetect::WBDetector`, `cv2.dpm.DPMDetector`, `cv::dnn_objdetect::InferBbox` | Domain-specific, legacy and unbound detector helpers remain deferred. Reuse the standard detector adapter when a pinned inspection model needs equivalent behavior. | — |
| `cv2.line_descriptor.BinaryDescriptor`, `cv2.line_descriptor.BinaryDescriptorMatcher`, `cv2.line_descriptor.DrawLinesMatchesFlags`, `cv::line_descriptor::EDLineDetector`, `cv::line_descriptor::EDLineParam`, `cv2.line_descriptor.KeyLine`, `cv2.line_descriptor.LSDDetector`, `cv2.line_descriptor.LSDParam`, `cv2.line_descriptor.drawKeylines`, `cv2.line_descriptor.drawLineMatches` | Group line extraction and descriptors with matched geometry. Line-based registration needs a separate alignment fixture. | S4-46, S4-99 |
| `cv2.wechat_qrcode.WeChatQRCode` | Qualify local models and known small or low-contrast symbols through the existing code result contract. | S4-25 |

#### Restoration and segmentation

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv::alphamat::infoFlow` | Trimap matting needs a qualified backend. The installed wheel has no direct binding. | S4-60 |
| `cv2.bioinspired.Retina`, `cv2.bioinspired.TransientAreasSegmentationModule` | Group transient-area and retina motion output with warmup, reset and exposure-change checks. | S4-54, S4-61 |
| `cv2.bioinspired.RetinaFastToneMapping` | Verify tone mapping on mixed illumination separately from stateful video processing. | S4-98 |
| `cv2.ft.createKernel`, `cv2.ft.createKernel1`, `cv2.ft.filter`, `cv2.ft.inpaint`, `cv2.ft.FT02D_FL_process`, `cv2.ft.FT02D_FL_process_float`, `cv2.ft.FT02D_components`, `cv2.ft.FT02D_inverseFT`, `cv2.ft.FT02D_iteration`, `cv2.ft.FT02D_process`, `cv2.ft.FT12D_components`, `cv2.ft.FT12D_createPolynomMatrixHorizontal`, `cv2.ft.FT12D_createPolynomMatrixVertical`, `cv2.ft.FT12D_inverseFT`, `cv2.ft.FT12D_polynomial`, `cv2.ft.FT12D_process` | Keep transforms and kernels internal to restoration. Qualify support-mask propagation separately from image-only fill. | S4-70, S4-100 |
| `cv2.intensity_transform.autoscaling`, `cv2.intensity_transform.contrastStretching`, `cv2.intensity_transform.gammaCorrection`, `cv2.intensity_transform.logTransform` | These variants follow the typed intensity transform contract. | S1-17 |
| `cv2.intensity_transform.BIMEF`, `cv2.intensity_transform.BIMEF2` | Qualify automatic and explicit exposure settings with the low-light fixture. | S4-73 |
| `cv2.hfs.HfsSegment` | Keep CPU and accelerator choices under the same image segmentation contract. | S4-71 |

Pinned sources are [classical ML](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/ml/include/opencv2),
[tracking](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/tracking/include/opencv2), [silhouette tracking](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/rapid/include/opencv2),
[object detection](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/xobjdetect/include/opencv2),
[legacy DNN detection](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/dnn_objdetect/include/opencv2), [deformable part models](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/dpm/include/opencv2),
[line descriptors](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/line_descriptor/include/opencv2), [WeChat QR](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/wechat_qrcode/include/opencv2),
[matting](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/alphamat/include/opencv2), [retina](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/bioinspired/include/opencv2),
[fuzzy transforms](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/fuzzy/include/opencv2), [intensity transforms](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/intensity_transform/include/opencv2),
and [hierarchical segmentation](https://github.com/opencv/opencv_contrib/tree/5.0.0/modules/hfs/include/opencv2).

### Native color calibration and image-profile resampling crosswalk

This audit maps 48 Python names from three pinned headers, including
color-model methods and the chart interface previously listed by class.
Image-profile resampling stays tied to an explicit image grid and sample pitch.
General-purpose signal processing stays outside the graph checklist.

| API | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.ccm.ColorCorrectionModel`, `cv2.ccm.ColorCorrectionModel.compute`, `cv2.ccm.ColorCorrectionModel.setCcmType`, `cv2.ccm.ColorCorrectionModel.setDistance`, `cv2.ccm.ColorCorrectionModel.setLinearization`, `cv2.ccm.ColorCorrectionModel.setLinearizationGamma`, `cv2.ccm.ColorCorrectionModel.setLinearizationDegree`, `cv2.ccm.ColorCorrectionModel.setSaturatedThreshold`, `cv2.ccm.ColorCorrectionModel.setWeightsList`, `cv2.ccm.ColorCorrectionModel.setWeightCoeff`, `cv2.ccm.ColorCorrectionModel.setInitialMethod`, `cv2.ccm.ColorCorrectionModel.setMaxCount`, `cv2.ccm.ColorCorrectionModel.setEpsilon` | Keep matrix form, linearization, fit weighting and convergence controls in model fitting. | S4-11, S4-101 |
| `cv2.ccm.ColorCorrectionModel.setColorSpace`, `cv2.ccm.ColorCorrectionModel.setRGB`, `cv2.ccm.gammaCorrection` | Separate input channel order, reference encoding and output color space. Gamma is a typed intensity transform. | S1-17, S4-11, S4-35, S4-101 |
| `cv2.ccm.ColorCorrectionModel.getColorCorrectionMatrix`, `cv2.ccm.ColorCorrectionModel.getLoss`, `cv2.ccm.ColorCorrectionModel.getSrcLinearRGB`, `cv2.ccm.ColorCorrectionModel.getRefLinearRGB`, `cv2.ccm.ColorCorrectionModel.getMask`, `cv2.ccm.ColorCorrectionModel.getWeights` | Expose useful fit evidence and distinguish it from errors measured on independent chart captures. | S4-11, S4-102 |
| `cv2.ccm.ColorCorrectionModel.correctImage`, `cv2.ccm.ColorCorrectionModel.read`, `cv2.ccm.ColorCorrectionModel.write` | Model persistence and typed correction follow the application contract. Native free read and write helpers stay internal. | S4-35, S4-102 |
| `cv2.mcc.CChecker`, `cv2.mcc.CChecker.getBox`, `cv2.mcc.CChecker.getCenter`, `cv2.mcc.CChecker.getColorCharts`, `cv2.mcc.CChecker.getChartsRGB`, `cv2.mcc.CChecker.getChartsYCbCr`, `cv2.mcc.CChecker.getCost`, `cv2.mcc.CChecker.getTarget` | Chart geometry and measurements stay with ordered patch records and independent verification. | S4-34, S4-102 |
| `cv2.mcc.CCheckerDetector`, `cv2.mcc.DetectorParametersMCC`, `cv2.mcc.CCheckerDetector.getBestColorChecker`, `cv2.mcc.CCheckerDetector.getListColorChecker`, `cv2.mcc.CCheckerDetector.getColorChartType`, `cv2.mcc.CCheckerDetector.setColorChartType`, `cv2.mcc.CCheckerDetector.getDetectionParams`, `cv2.mcc.CCheckerDetector.setDetectionParams`, `cv2.mcc.CCheckerDetector.process`, `cv2.mcc.CCheckerDetector.processWithROI`, `cv2.mcc.CCheckerDetector.getUseDnnModel`, `cv2.mcc.CCheckerDetector.setUseDnnModel`, `cv2.mcc.CCheckerDetector.draw` | Native ROI, backend and review choices follow the same chart detection workflow. | S4-34 |
| `cv2.mcc.CCheckerDetector.getRefColors` | Distinguish built-in references from measured color samples and their declared encoding. | S4-11, S4-101–S4-102 |
| `cv2.signal.resampleSignal` | The native wrapper accepts a float32 row and integer rates of at least 1000. The pinned source applies its FIR only when input rate is less than twice output rate. Verify alias rejection or qualify another resampler for inspection profiles. | S1-173 |

Pinned sources are [color correction](https://github.com/opencv/opencv/blob/5.0.0/modules/photo/include/opencv2/photo/ccm.hpp),
[chart detection](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/include/opencv2/objdetect/mcc_checker_detector.hpp), and
[profile resampling](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/signal/include/opencv2/signal/signal_resample.hpp).

### Inspection metrology and residual image API review

This pass checks the remaining top-level image names in the pinned wheel against
existing graph tools and the inspection backlog. Hough accumulator wrappers are
already used by the polar-line and circle blocks. Separate metrology rows cover
measurements that need new outputs and validation.

| API or workflow | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.HoughLinesWithAccumulator`, `cv2.HoughCirclesWithAccumulator` | Shipped Hough tools already emit native vote or strength data for filtering and review. | S1-34 |
| `cv2.fisheye.initUndistortRectifyMap`, `cv2.fisheye.undistortImage` | Map construction and direct resampling follow the existing declared fisheye camera and rectification contracts. Fitting the profile remains separate. | S3-04, S3-09, S3-14 |
| `cv2.rectangleIntersectionArea` | Axis-aligned intersection is a geometry helper within footprint overlap and detection evaluation. | S1-33, S2-23 |
| `cv2.fitLine` | The image fit supplies individual lines. Line-pair metrology now measures frame-aware corners, angles and gaps; spatial-axis fitting remains a separate planned workflow. | S1-33, S1-174, S3-129 |
| `cv2.calibrateCameraExtended` | Shipped fit diagnostics support capture-set assessment. Add coverage, pose diversity and repeatability checks on the set. | S3-02, S3-127 |
| `cv2.projectPoints`, `cv2.projectPointsSepJ` | Use projection derivatives or perturbation trials within calibrated uncertainty measurement. Preserve covariance assumptions and output units. | S3-09, S3-128 |
| `Edge-spread and spatial-frequency response` | `inspect.slanted_edge` derives selected ESF/LSF/SFR and MTF50/10 on the current pixel grid, with explicit sample pitch, windowing, derivative correction, Nyquist bounds and unsupported-edge reports. Qualified against analytic Gaussian edges; no ISO compliance claim. | S1-175 |
| `Custom BRISK sampling layouts` | Treat custom radii, sample counts and comparison ordering as extensions of the existing descriptor contract. Qualify persisted settings and matching compatibility before claiming support for the extension. | S1-43 |
| `Array and feature record types` | Mat, rotated rectangles, keypoints, matches and file-storage objects remain representations inside graph tools. | — |
| `Python bootstrap, HighGUI and platform utilities` | Keep package initialization, windowing, timings and hardware controls under packaging and runtime qualification. | — |

Sources include the pinned [image processing header](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/include/opencv2/imgproc.hpp),
[2D geometry header](https://github.com/opencv/opencv/blob/5.0.0/modules/geometry/include/opencv2/geometry/2d.hpp),
[3D geometry header](https://github.com/opencv/opencv/blob/5.0.0/modules/geometry/include/opencv2/geometry/3d.hpp),
and [calibration header](https://github.com/opencv/opencv/blob/5.0.0/modules/calib/include/opencv2/calib.hpp).
The metrology items are planned workflows composed from these primitives.
Native solver parameter deviations are retained with their parameter order and
fixed-parameter flags. Measurement uncertainty needs a declared joint covariance
or empirically checked perturbation model.

### Camera model compatibility audit

Projection support does not establish pose-estimation support for the same
lens. The camera record accepts pinhole, fisheye and omnidirectional models.
The graph's projection and point-undistortion paths dispatch by lens, while
object and relative pose currently require pinhole input.

| API or workflow | Graph disposition | Checklist IDs |
| --- | --- | --- |
| `cv2.fisheye.solvePnP`, `cv2.fisheye.solvePnPRansac` | Qualify wide-angle object pose on known 3D targets, including outliers. Evaluate the resulting pose through the original lens model, with declared pixel residuals and supported ray domain. | S3-130 |
| `cv2.findEssentialMat`, `cv2.recoverPose` | Qualify the independent-camera overload or per-lens normalized correspondences. Retain each image grid and calibration; preserve the unitless translation direction until scale is anchored. | S3-131 |
| `cv::sfm::triangulatePoints` | Extend existing reconstruction work to three or more supplied or estimated cameras, with missing observations and persistent landmark IDs. Keep two-view baseline coverage separate. | S3-12, S3-33 |
| `cv2.minEnclosingCircle` | The native enclosing circle bounds its input points. inspect.fit_circle_arc separately performs normalized CPU geometric least squares with optional RANSAC, SVD refinement, radial residuals, observed angular envelopes and conditional uncertainty. Short-arc, conditioning and point-order checks have independent truth fixtures; no native fitCircle binding is claimed. | S1-33, S1-134 |

The pinned [3D geometry header](https://github.com/opencv/opencv/blob/5.0.0/modules/geometry/include/opencv2/geometry/3d.hpp)
declares the fisheye solvers and separate-camera essential/recovery overloads.
The [multi-view triangulation header](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/sfm/include/opencv2/sfm/triangulation.hpp)
takes observations and projection matrices across views. Its Python binding is
absent from the pinned wheel. These are extensions of the existing pose and
reconstruction contracts; no additional generic matrix tools are needed.

### Remaining inspection workflows

| Family | Roadmap disposition | Checklist IDs |
| --- | --- | --- |
| `Image I/O and reconstruction details` | Cover multipage buffer ports, HDR motion rejection, ternary morphology, and sparse contour extraction as distinct graph workflows. | S1-149–S1-152 |
| `Camera calibration and image tracking` | Track release-object and metadata-aware board calibration plus sparse RLOF. | S3-109–S3-110, S4-84 |
| `Point-cloud metrology and large-scan processing` | Cover matched-landmark transforms, stream/tile processing, recorded scan replay, cross-sections, datum measurements and calibrated range images. Qualify cylinder and bore fits with a tested solver. | S3-101, S3-103–S3-106, S3-112, S3-115 |
| `Mesh quality and processing` | Cover texture projection, topology/measurement checks and bounded simplification. Qualify a tested equivalent where OpenCV supplies no direct mesh-processing primitive. | S3-102, S3-107–S3-108 |

### Backlog decisions

| Backlog family | Roadmap disposition | Checklist IDs |
| --- | --- | --- |
| `Illumination, surface defects and depth cleanup` | Count reusable operations for cast shadows, chromatic fringe correction, depth confidence and boundary cleanup. Package application recipes and qualify them with part-specific data in Stage 5. | S1-55–S1-56, S1-116, S1-145–S1-146, S3-19, S3-73 |
| `Multi-instance picking and occlusion` | Track 2D visible support and calibrated 3D visibility and matching separately from the delivered picking job. | S1-61, S3-24, S3-52, S3-91 |
| `Defect and production validation` | Production watershed, color-chart qualification, direct-part-mark fixtures, barcode grading, and defect recipe scoring belong to application validation. Reusable image tools remain tracked in their stages. | — |
| `Source timing and real equipment` | Acquisition owns independent sources and hardware sync. Industrial integration owns controller and robot connectivity. Keep recorded vision replay and calibrated geometry in scope. | S1-09–S1-12, S1-123, S3-31–S3-32, S3-62 |
| `scikit-image texture and restoration extensions` | Keep Frangi, Sato, local binary patterns and co-occurrence under the separate scikit-image tools. OpenCV ridge and texture alternatives retain their own rows. | S1-77, S4-05 |

## Maintain the baseline

When changing OpenCV versions, compare source modules and the binding inventory
against this map. Add or update dispositions for changed families. When adding a
roadmap capability, give it a stable ID and a measurable result, then add its
owning family reference before refreshing the coverage table. A new item
needs a distinct inspection result or qualification check. Treat overloads,
aliases and native helper methods as part of their owning graph workflow.
Callable-level audit boundaries are the pinned headers named in each
crosswalk. Use the module and subgroup map to review additional APIs when
a new inspection workflow or a supported build exposes a gap. The module map
records backend, platform and domain boundaries.
Model layer internals and general-purpose numerics stay outside the graph
checklist unless a distinct inspection workflow needs them. Reuse an existing
ID for equivalent operations.

Use exact `cv2.*` names for exposed Python symbols in crosswalk table keys.
Use C++ names for source APIs that lack a binding. The
[roadmap](roadmap.md) gives implementation order. The checklist describes
graph coverage, while module mapping describes scope.
