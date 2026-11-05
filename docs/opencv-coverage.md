# OpenCV coverage backlog

This inventory compares the built-in graph registry with the public Python
bindings in the pinned OpenCV 5 contrib wheel. Priorities reflect reuse across
industrial vision jobs and the workflows in the [roadmap](roadmap.md). They
are qualitative estimates. The [block parameter reference](block-parameters.md)
lists the tools that can be placed on a graph today.

## P1 — Common tools and workflow gaps

| Rank | Outstanding capability | OpenCV primitives and graph work |
| --- | --- | --- |
| 1 | Separate touching parts and isolate foreground | Marker-controlled watershed, distance-peak seeds, GrabCut from rectangle or mask, flood fill, hole fill, morphological reconstruction and skeleton thinning. Existing threshold, morphology, distance transform and components are building blocks. |
| 2 | Reference comparison and defect scoring | SSIM and local similarity map, PSNR/MSE, histogram comparison and backprojection, registered reference subtraction, residual components and fail thresholds. Absolute difference and histograms already exist. |
| 3 | Multi-result template search and pick selection | Response-map peaks from `matchTemplate`, score thresholds, non-maximum suppression, subpixel peaks and pose-relative clearance. Add distinct visible-feature coverage, overlap and depth occlusion evidence, and deterministic pick ranking. Existing trainable matchers and collection filters cover parts of this pipeline. |
| 4 | Calipers, widths, contours and shape fitting | Intensity profiles along pose-relative calipers, paired edges, width/gap/discontinuity along a path, `fitLine`, `fitEllipse*`, `minEnclosingCircle`, convex hull/defects, `pointPolygonTest` and contour-to-contour `matchShapes`. Several descriptors are computed inside existing detectors. Dedicated fitting blocks should accept points or collections and report residuals. |
| 5 | File video and replay | Video file/stream sources, decode timestamps, seek, frame ranges and extraction, codec/container selection, video writing and deterministic end-of-stream. Extend image-camera acquisition with timed replay and independent source readiness. |
| 6 | Industrial text and code workflows | OCR text detection, recognition, character rules and training/recipe assets. Add verified Data Matrix and other required symbologies through a suitable backend, and a separate grading workflow. Basic QR and OpenCV linear-code decoding are available. |
| 7 | Color tolerance and coating checks | ROI color statistics, Lab color difference including ΔE00, color-chart calibration, correction matrix and color-class decisions. Existing histogram and color balance remain useful inputs. |
| 8 | Motion and conveyor tracking | MOG2/KNN alternatives to existing background memory, CSRT/KCF, detection association, encoder/time pairing, coordinate propagation and stale-track expiry. Existing sparse LK and Farneback flow provide motion primitives. |
| 9 | Robot and calibration workflows | ChArUco board observations, identifiable origins, independent reprojection validation, hand–eye and robot-world/hand–eye calibration, frame composition tied to acquisition timestamps, held-out target checks and calibration revision selection. PnP and basic ArUco pose are already available. |

## P2 — Common extensions

| Family | Outstanding capability |
| --- | --- |
| Image restoration | Fast non-local means, temporal NLM, TV-L1 denoising, Telea/Navier–Stokes inpainting, edge-preserving/guided/domain-transform filters, deconvolution/deblur, shadow and illumination correction. |
| Registration and stitching | ECC and multiscale ECC, masked intensity alignment, camera motion stabilization, sparse alignment diagnostics, panorama and inspection mosaic stitching, seam/exposure correction and blending. |
| Feature detection and correspondence | Standalone SIFT, AKAZE, BRISK, KAZE, MSER and descriptor adapters; selectable BF/FLANN KNN, radius matching, cross-check and geometric inlier scoring. ORB/FAST and SIFT-based match pipelines already exist. |
| Shape and topology | Generalized Hough, contour approximation and hierarchy as separate operations, ellipse/arc and polygon models, rotated intersection, connected-component algorithm choices, shape context and Zernike-style descriptors where available. |
| Segmentation and texture | SLIC/SEEDS/LSC superpixels, meanshift, k-means, graph segmentation, selective search, ridge/Frangi-style response, local binary patterns, co-occurrence matrices and Gabor banks. |
| Spectral and intensity processing | DFT/IDFT, DCT/IDCT, spectrum/filter masks, frequency-domain correlation, per-channel LUTs, log/exp/power, clipping, generalized percentile/statistics, integral images and tilt-aware background removal. |
| Image metrics | Full-reference SSIM/PSNR/MAE/MSE, quality maps and reductions, BRISQUE/other no-reference scores, perceptual image hashes, Earth Mover’s Distance and defect score calibration. See P1 for the first comparison tools. |
| Motion and background | DIS, Dual TV-L1, DeepFlow and RLOF where bound, foreground masking and adaptation, optical-flow consistency and visualization, Kalman filtering and tracking across missing detections. |
| Calibration models and target tooling | Fisheye/omnidir, circle-grid options, distortion visualization, undistort/rectify map export, calibration uncertainty, multi-camera extrinsics and sparse bundle adjustment where supported. |
| Depth and stereo | Disparity WLS and left–right consistency, depth filling/edge-aware filtering, confidence planes, time-of-flight depth cleanup, alternative stereo algorithms and dense calibration evaluation. |
| Point cloud and reconstruction | Surface/mesh output, organized segmentation and plane fitting, feature registration, multiple hypotheses and fitness ranking, occlusion/z-buffer visibility, SfM and more complete multi-frame pose/mesh workflows. Existing ICP, PPF, depth/point-cloud conversion and TSDF cover the baseline. |
| DNN task adapters | Multi-input model binding, dynamic shapes and batching, detection/rotated detection, NMS, semantic/instance masks, keypoints and OCR postprocessing, confidence calibration, portable preprocessing and CPU-vs-accelerator validation. Existing ONNX inference, classification and embeddings provide the base. |
| Classical ML | Train/predict/export with k-means, SVM, KNearest, trees/forests, boosting, EM and other `ml` estimators. Add feature-table preparation and train/validation split. |
| Media and image formats | In-memory encode/decode, TIFF/multipage files, metadata and EXIF handling, Bayer/demosaic options, YUV and other sensor formats, codecs/compression settings and additional numerical formats. |

## P3 — Specialized families

| Family | Outstanding capability |
| --- | --- |
| Photo and exposure | Exposure alignment/merge, HDR/tone mapping, seamless clone, decolorization, stylization, motion deblur, `xphoto` white-balance variants and learning-based super-resolution. |
| Recognition | WeChat QR, QR encoding, multiple-code layout, cascade/HOG, face detection/recognition, saliency, line descriptors, image retrieval/vocabulary and OpenCV-specific OCR variants. |
| Sequence | Additional tracker families, background subtractors in `bgsegm`, motion history and advanced video stabilization. |
| Geometric utilities | Expanded robust estimators, graph matching, rectification variants, affine-invariant feature variants, panorama detail steps and coordinate-space diagnostics. |
| Diagnostics and visualization | Color maps and display overlays as reusable blocks, match/inlier overlays, density maps, 3D axes and camera frusta, disparity confidence views and inspection measurement plots. |
| Contrib and domain tools | Fuzzy and bioinspired processing, structured-light modes beyond Gray code, saliency, detailed image hashing and optional SfM, mesh/visualization and hardware-specific modules where the Python build supplies them. |

## P4 — Infrastructure and build-dependent features

- CUDA/OpenCL/UMat acceleration, hardware decoder/encoder backends, camera
  device controls, DNN targets, and alternate runtimes require explicit build
  probes and end-to-end measurements.
- OpenCV.js/WebAssembly and compiled ToolBlocks belong to the
  [execution roadmap](opencv-backend.md).
- Drawing helpers, matrix constructors, getters/setters, serializers, backend
  controls and overloaded aliases generally belong inside higher-level nodes.
  Wheel bindings do not establish device support or calibration accuracy.

## Exact callable inventory

The [binding inventory](opencv-bindings.tsv) includes functions, classes and
public methods declared on those classes. Filter by priority and
`no static reference` to find candidates. `literal reference` means a
`cv2.…` expression occurs in application source. `method name only` means
the method name occurs somewhere in application source and may have a
different owner. `alias_of` identifies bindings that point to the same
Python object and the corresponding methods. The scan cannot decide whether a
reference is internal, configurable on a graph, dynamically resolved, or
equivalent to a tool already built from other primitives. The prioritized
tables above make that distinction for the main workflows.

Run `python scripts/opencv_binding_inventory.py` from the repository using its
pinned environment to refresh the TSV. Add `--check` to verify it. Constants,
private symbols and Python helper modules are omitted. OpenCV features absent
from this wheel must be evaluated against the intended distribution separately.
See the [OpenCV 5 documentation](https://docs.opencv.org/5.0/).

## Available code readers

**Read QR codes** supports multi, single and curved QR modes. **Read linear
barcodes** exposes native detector scales, gradient/downsampling thresholds
and optional super-resolution model files. Both take uint8 images with
optional masks and regions. Keep the full code and its quiet zone inside the
selection. Both return decoded text, symbology, original image-space corners,
center, count and an optional overlay. A minimum read count can fail the block
and report a reason. The automated fixtures decode QR and EAN-13. Supported
linear symbologies and decode behavior depend on the OpenCV implementation.
See the [OpenCV barcode documentation](https://docs.opencv.org/5.0/main_modules/objdetect_barcode.html).
