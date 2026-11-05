# OpenCV coverage backlog

This inventory compares the built-in graph registry with the public Python
bindings in the pinned OpenCV 5 contrib wheel. Priorities reflect reuse across
industrial vision jobs and the workflows in the [roadmap](roadmap.md). They
are qualitative estimates. The [block parameter reference](block-parameters.md)
lists the tools that can be placed on a graph today.

## P1 — Common tools and workflow gaps

| Rank | Outstanding capability | OpenCV primitives and graph work |
| --- | --- | --- |
| 1 | Separate touching parts and isolate foreground | The basic workflow is available, including distance-peak seed generation, watershed and morphological reconstruction. Add a reusable touching-parts job with calibration and validation fixtures. |
| 2 | Reference comparison and defect scoring | The basic workflow is available, including histogram backprojection and a reusable registered defect inspection. Add validated defect-specific inspection recipes and lighting normalization. |
| 3 | Multi-result template search and pick selection | Subpixel template peaks, pose-relative clearance, distinct visible-feature coverage and depth occlusion evidence, and deterministic pick ranking. Fixed-scale multi-result template matching now supplies ranked peaks, overlap suppression and poses. Trainable matchers and collection filters cover other parts of this pipeline. |
| 4 | Calipers, widths, contours and shape fitting | Intensity profiles along pose-relative calipers, paired edges, width/gap/discontinuity along a path, convex hull/defects, `pointPolygonTest` and contour-to-contour `matchShapes`. Dedicated line, ellipse and minimum enclosing circle fits now accept points or a collection item and report residuals. |
| 5 | File video and replay | Video file/stream sources, decode timestamps, seek, frame ranges and extraction, codec/container selection, video writing and deterministic end-of-stream. Extend image-camera acquisition with timed replay and independent source readiness. |
| 6 | Industrial text and code workflows | OCR text detection, recognition, character rules and training/recipe assets. Add verified Data Matrix and other required symbologies through a suitable backend, and a separate grading workflow. Basic QR and OpenCV linear-code decoding are available. |
| 7 | Color tolerance and coating checks | ROI color statistics, Lab color difference including ΔE00, color-chart calibration, correction matrix and color-class decisions. Existing histogram and color balance remain useful inputs. |
| 8 | Motion and conveyor tracking | MOG2/KNN alternatives to existing background memory, CSRT/KCF, detection association, encoder/time pairing, coordinate propagation and stale-track expiry. Existing sparse LK and Farneback flow provide motion primitives. |
| 9 | Robot and calibration workflows | ChArUco board observations, identifiable origins, independent reprojection validation, hand–eye and robot-world/hand–eye calibration, frame composition tied to acquisition timestamps, held-out target checks and calibration revision selection. PnP and basic ArUco pose are already available. |

## P2 — Common extensions

| Family | Outstanding capability |
| --- | --- |
| Image restoration | Temporal NLM, TV-L1 denoising, edge-preserving/guided/domain-transform filters, deconvolution/deblur, shadow and illumination correction. Single-frame non-local means and Telea/Navier–Stokes inpainting are available. |
| Registration and stitching | Camera motion stabilization, sparse alignment diagnostics, panorama and inspection mosaic stitching, seam/exposure correction and blending. Single-scale and multiscale ECC support intensity alignment with masks. |
| Feature detection and correspondence | Standalone SIFT, AKAZE, BRISK, KAZE, MSER and descriptor adapters; selectable BF/FLANN KNN, radius matching, cross-check and geometric inlier scoring. ORB/FAST and SIFT-based match pipelines already exist. |
| Shape and topology | Generalized Hough, contour approximation and hierarchy as separate operations, ellipse/arc and polygon models, rotated intersection, connected-component algorithm choices, shape context and Zernike-style descriptors where available. |
| Segmentation and texture | SLIC/SEEDS/LSC superpixels, meanshift, k-means, graph segmentation, selective search, ridge/Frangi-style response, local binary patterns, co-occurrence matrices and Gabor banks. |
| Spectral and intensity processing | DFT/IDFT, DCT/IDCT, spectrum/filter masks, frequency-domain correlation, per-channel LUTs, log/exp/power, clipping, generalized percentile/statistics, integral images and tilt-aware background removal. |
| Image metrics | MAE and additional map reductions, BRISQUE and other no-reference scores, perceptual image hashes, Earth Mover’s Distance and defect score calibration. |
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


## Available segmentation and comparison tools

**Seeded flood fill** returns a mask, area and bounds. It supports fixed
or floating intensity tolerances, 4/8-connectivity and a mask or region
that acts as a barrier. **Marker-controlled watershed** accepts a uint8
seed mask or a nonnegative int32 label image. It returns labels, a boundary
mask and a mask for one selected label or all foreground labels.
**Watershed seeds from distance peaks** converts a binary foreground mask
into int32 markers. It uses label 1 for known background, label 0 for
unassigned pixels and labels 2 and up for foreground peaks. Connect
`markers` to **Marker-controlled watershed**, select `labels` as the
marker mode and set the background label to 1. Connect the original
grayscale or color image to the watershed image input.
See the [OpenCV distance-transform watershed tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/imgtrans/distance_transformation/distance_transform.html).
**Morphological reconstruction** propagates a monochrome marker under
an upper image bound by dilation or above a lower bound by erosion.
It reports the number of geodesic steps and withholds the image when
the iteration limit is reached before convergence.
**GrabCut foreground** accepts a rectangle, a binary probable-foreground mask
or native four-state markers. It initializes a new model for each activation.
**Thin binary mask** exposes Zhang–Suen and Guo–Hall skeletonization.
**Fill holes in a binary mask** fills enclosed background components, with an
optional maximum hole area and configurable background connectivity.

**Compare image quality** compares two registered images using SSIM, PSNR or
MSE and an optional pass threshold. A region or mask selects pixels for the
score. SSIM uses neighborhoods from the full images, so nearby unselected
pixels can affect a selected pixel's similarity. SSIM and MSE return full
maps. The score uses equal weighting across selected pixels and channels.


## Available repeated template matching

**Find template instances** searches for fixed-scale, fixed-orientation
copies with a normalized matching method, minimum score, result limit,
overlap suppression and optional mask/region inputs. It returns ranked
bounds, centers, scores and a list of center poses. The pose list connects
directly to **Inspect each pose in parallel**. Open the [25-part example](../examples/template_instances.json)
or run it with `opencv-dataflow-run examples/template_instances.json`.


## Available histogram and shape measurements

**Compare histograms** accepts matching Histogram outputs and offers all six
OpenCV comparison metrics, optional count normalization, per-channel results
and a threshold on the mean or worst channel. Use identical bin edges and
ranges in both source histograms. **Histogram backprojection** maps one
modeled channel to a float32 image of per-pixel weights. It supports an
optional region or mask, uniform histogram bins, and peak, probability or
raw-count weighting. Convert the reference and sample channels to the same
numeric scale before comparing or backprojecting.
See the [OpenCV backprojection tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/histograms/back_projection/back_projection.html).

**Fit line, ellipse or enclosing circle** fits points or one detected object.
It reports image-space geometry and root mean squared distance to the fitted
boundary, with an optional residual tolerance. Circle mode returns the
smallest enclosing circle.


## Available alignment and restoration

**Align images (ECC)** registers a sample to a reference with single-scale or
multiscale enhanced correlation. It exposes translation, rigid, affine and
homography models, optional masks for both frames, an optional initial
transform and a minimum coefficient. It emits the aligned image and the
sample-to-reference transform. The inverse mapping is available on its
geometry output.

Import the [Registered defect inspection ToolBlock](../examples/registered_defect_inspection.toolblock.json)
to align a sample to a known-good reference, threshold the absolute
difference and report connected defects. Its optional region is in reference
coordinates. The included settings use translation, a residual threshold of
35 intensity levels and a minimum defect area of 12 pixels. Tune these
settings in the nested graph for your images.

**Non-local means denoise** offers channel-based and luminance/chroma modes.
A region or mask limits where denoised pixels are written. **Inpaint selected
pixels** reconstructs a masked area from neighboring pixels with Telea or
Navier–Stokes.
