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
| `ptcloud` | Inspection cloud import, transforms, visibility, normals, sampling, segmentation, mesh and fusion. Vendor acquisition and unrelated GIS work have separate scope. | S3-20–S3-27, S3-50–S3-53, S3-56–S3-60, S3-64, S3-71 |
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

### Remaining inspection workflows

| Family | Roadmap disposition | Checklist IDs |
| --- | --- | --- |
| `Image I/O and reconstruction details` | Cover multipage buffer ports, HDR motion rejection, ternary morphology, and sparse contour extraction as distinct graph workflows. | S1-149–S1-152 |
| `Camera calibration and image tracking` | Track release-object and metadata-aware board calibration plus sparse RLOF. | S3-109–S3-110, S4-84 |
| `Point-cloud metrology and large-scan processing` | Cover matched-landmark transforms, stream/tile processing, recorded scan replay, cross-sections, datum measurements and calibrated range images. | S3-101, S3-103–S3-106, S3-112 |
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
Use exact `cv2.*` names for exposed Python symbols in crosswalk table keys.
Use C++ names for source APIs that lack a binding. The
[roadmap](roadmap.md) gives implementation order. The checklist describes
graph coverage, while module mapping describes scope.
