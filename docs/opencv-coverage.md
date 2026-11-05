# OpenCV coverage backlog

This inventory compares the built-in graph registry with the public Python
bindings in the pinned OpenCV 5 contrib wheel. Priorities reflect reuse across
industrial vision jobs and the workflows in the [roadmap](roadmap.md). They
are qualitative estimates. The [block parameter reference](block-parameters.md)
lists the tools that can be placed on a graph today. The [roadmap](roadmap.md#vision-tool-priorities)
sets the order for development across OpenCV core, ML/DNN, 3D, contrib and
application-specific ToolBlocks. The P1–P3 rankings below prioritize work
within those stages.

## P1 — Common tools and workflow gaps

| Rank | Outstanding capability | OpenCV primitives and graph work |
| --- | --- | --- |
| 1 | Separate touching parts and isolate foreground | [Watershed separation](../examples/watershed/README.md) includes distance peaks, editable point seeds, mask seeds, and a synthetic touching-parts example. Production calibration and validation fixtures remain open. |
| 2 | Reference comparison and defect scoring | The basic workflow is available, including histogram backprojection and a reusable registered defect inspection. [Flat-field correction](../examples/flat_field/README.md) normalizes illumination using bright and optional dark references. Validated defect-specific inspection recipes remain open. |
| 3 | Multi-result template search and pick selection | Explicit visible-feature evidence and depth-based occlusion checks remain open. Multi-result matching and image-plane pick selection support score, pose angle, clearance and deterministic downstream ranking. [Subpixel refinement](../examples/subpixel_template/README.md) is available for template centers and poses. |
| 4 | Calipers, widths, contours and shape fitting | Curved path sampling, contour-profile tolerances and physical-unit tolerance recipes. Pose-relative paired-edge calipers now measure width, center offset and gaps across straight parallel scans. Hull, contour similarity, [contour approximation](../examples/contour_approximation/README.md), point-to-contour distance and dedicated shape fits are available. |
| 5 | File video and replay | Timed playback, live network streams, independent source readiness and deeper codec validation. Local video replay and writing are available with frame ranges, stride, frame tickets, timestamp provenance and stop/loop modes. See [video file workflows](video-files.md). |
| 6 | Industrial text and code workflows | Trainable OCR workflow assets, qualification of direct part marks on production images, and a separate barcode grading workflow. OCR and decoded-text checks are available alongside OpenCV QR/linear readers and optional ZXing Data Matrix and multi-format decoding. See [OCR workflows](ocr.md). |
| 7 | Color tolerance and coating checks | Production color-chart qualification remains open. [Multi-reference Lab classification](../examples/lab_color_classification/README.md) is available. Color-chart detection, fitted correction matrices, selected Lab statistics and CIE76/CIEDE2000 tolerance checks are available. See [color measurement](color-measurement.md). |
| 8 | Motion and conveyor tracking | PLC/camera hardware synchronization, timestamp-window pairing, calibrated world coordinates and production validation. Multi-part association and frame-latched encoder prediction support tracking beyond the camera view in image coordinates. CSRT/KCF track seeded boxes, and MOG2/KNN, sparse LK and Farneback provide motion primitives. |
| 9 | Robot and calibration workflows | ChArUco board observations, identifiable origins, independent reprojection validation, hand–eye and robot-world/hand–eye calibration, frame composition tied to acquisition timestamps, held-out target checks and calibration revision selection. PnP and basic ArUco pose are already available. |

## P2 — Common extensions

| Family | Outstanding capability |
| --- | --- |
| Image restoration | Temporal NLM, TV-L1 denoising, deconvolution/deblur, and shadow correction. Single-frame non-local means, Telea/Navier–Stokes inpainting, [guided filtering](../examples/guided_filter/README.md), and [domain transform filtering](../examples/domain_transform_filter/README.md) are available. |
| Registration and stitching | Camera motion stabilization, sparse alignment diagnostics, panorama and inspection mosaic stitching, seam/exposure correction and blending. Single-scale and multiscale ECC support intensity alignment with masks. |
| Feature detection and correspondence | Custom BRISK patterns remain open. [Descriptor radius matching](../examples/descriptor_radius/README.md) retains multiple candidates per keypoint within a distance limit. [MSER](../examples/mser_regions/README.md) detects stable grayscale and color regions. [Standard BRISK](../examples/brisk_registration/README.md) supports Hamming matching. ORB/FAST and [standalone SIFT](../examples/sift_registration/README.md) are available. [AKAZE](../examples/akaze_registration/README.md) supports binary and float descriptors. ORB, SIFT, and AKAZE matching support ratio and cross-check modes, with homography inlier scoring downstream. [FLANN KD-tree matching](../examples/flann_registration/README.md) accepts SIFT, [standalone KAZE](../examples/kaze_registration/README.md), and float AKAZE descriptors. |
| Shape and topology | Generalized Hough, ellipse/arc and additional polygon models, rotated intersection, connected-component algorithm choices, shape context and Zernike-style descriptors where available. [Contour hierarchy](../examples/contour_hierarchy/README.md) exposes parent and child links, nesting depth, and holes. |
| Segmentation and texture | SLIC/SEEDS/LSC superpixels, meanshift, graph segmentation, selective search, ridge/Frangi-style response, local binary patterns, co-occurrence matrices and Gabor banks. [Pixel K-means](../examples/kmeans/README.md) is available. |
| Spectral and intensity processing | [DCT/IDCT](../examples/dct_round_trip/README.md) supports reversible monochrome transforms. [DFT/IDFT and radial filtering](../examples/fourier_filter/README.md) support band selection. [Per-channel intensity LUTs](../examples/intensity_lut/README.md) support uint8 and uint16. [Intensity transforms](../examples/intensity_transform/README.md) include log, exp, square root, power, and clipping. [Image arithmetic](../examples/image_arithmetic/README.md) combines two images or an image and a scalar. Custom spectrum masks, frequency-domain correlation, generalized percentile/statistics and tilted integral images remain open. [Local window statistics](../examples/local_statistics/README.md) use summed-area tables for rectangular mean, standard deviation, and sample-count maps. |
| Image metrics | MAE and additional map reductions, BRISQUE and other no-reference scores, perceptual image hashes, Earth Mover’s Distance and defect score calibration. |
| Motion and background | DIS, Dual TV-L1, DeepFlow and RLOF where bound, optical-flow consistency and visualization, Kalman filtering and tracking across missing detections. MOG2 and KNN provide adaptive foreground masking. |
| Calibration models and target tooling | Fisheye/omnidir, circle-grid options, distortion visualization, undistort/rectify map export, calibration uncertainty, multi-camera extrinsics and sparse bundle adjustment where supported. |
| Depth and stereo | Edge-aware depth filtering, time-of-flight depth cleanup, alternative stereo algorithms and dense calibration evaluation. Stereo BM/SGBM supports [WLS confidence and left–right consistency](../examples/stereo_confidence/README.md). [Depth hole filling](../examples/depth_hole_filling/README.md) is available. |
| Point cloud and reconstruction | Surface/mesh output, organized segmentation and plane fitting, feature registration, multiple hypotheses and fitness ranking, occlusion/z-buffer visibility, SfM and more complete multi-frame pose/mesh workflows. Existing ICP, PPF, depth/point-cloud conversion and TSDF cover the baseline. |
| DNN task adapters | Multi-input model binding, dynamic shapes and batching, detection/rotated detection, NMS, semantic/instance masks, keypoints and OCR postprocessing, confidence calibration, portable preprocessing and CPU-vs-accelerator validation. Existing ONNX inference, classification and embeddings provide the base. |
| Classical ML | Persist and reuse fitted K-means centers for prediction, and add train/predict/export with SVM, KNearest, trees/forests, boosting, EM and other `ml` estimators. Add feature-table preparation and train/validation split. [Numeric K-means fitting](../examples/kmeans/README.md) is available. |
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
**Watershed seeds from points** and **Watershed seeds from masks** construct the
same label map from manually or programmatically selected cores. The watershed
block includes a color preview and a per-label foreground area table.
See the [OpenCV distance-transform watershed tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/imgtrans/distance_transformation/distance_transform.html).
**Morphological reconstruction** propagates a monochrome marker under
an upper image bound by dilation or above a lower bound by erosion.
It reports the number of geodesic steps and withholds the image when
the iteration limit is reached before convergence.
**GrabCut foreground** accepts a rectangle, a binary probable-foreground mask
or native four-state markers. It initializes a new model for each activation.
**Thin binary mask** exposes Zhang–Suen and Guo–Hall skeletonization.
The optional scikit-image blocks expose Zhang, Lee, Guo–Hall, and medial-axis
skeletonization, plus length-based pruning of terminal branches. See the
[skeleton pruning example](../examples/skeleton_pruning/README.md).
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
directly to **Inspect each pose in parallel**. Open the [25-part example](../examples/template_instances/template_instances.json)
or run it with `opencv-dataflow-run examples/template_instances/template_instances.json`.


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

**Compare contours (Hu moments)** measures similarity between ordered point
lists or selected contour/blob collection items. **Contour hull and convexity
defects** reports hull vertices, solidity and concavity depths from a
whole-pixel contour. **Test points against contour** classifies points as
inside, on the edge or outside, and can filter them by signed distance to the
boundary. See the [OpenCV shape-analysis reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html).


## Available alignment and restoration

**Align images (ECC)** registers a sample to a reference with single-scale or
multiscale enhanced correlation. It exposes translation, rigid, affine and
homography models, optional masks for both frames, an optional initial
transform and a minimum coefficient. It emits the aligned image and the
sample-to-reference transform. The inverse mapping is available on its
geometry output.

Import the [Registered defect inspection ToolBlock](../examples/registered_defect_inspection/registered_defect_inspection.toolblock.json)
to align a sample to a known-good reference, threshold the absolute
difference and report connected defects. Its optional region is in reference
coordinates. The included settings use translation, a residual threshold of
35 intensity levels and a minimum defect area of 12 pixels. Tune these
settings in the nested graph for your images.

**Non-local means denoise** offers channel-based and luminance/chroma modes.
A region or mask limits where denoised pixels are written. **Inpaint selected
pixels** reconstructs a masked area from neighboring pixels with Telea or
Navier–Stokes.


## Available conveyor pick selection

**Select pick candidate** accepts the match records from shape, texture or
fixed-scale template matching. It projects a configured part rectangle and
clearance margin around each pose. Candidates can be rejected for a low
score, an out-of-range angle, a center outside the pick region, a clearance
rectangle outside the image, or interference with another detected part.
An optional uint8 mask adds obstacle evidence outside the candidate's own
footprint. Other supplied detections contribute obstacle footprints even
when their scores or angles make them ineligible for selection.

The block returns one pose, eligible count, per-candidate rejection reasons
and an optional overlay. Ranking uses the configured image-space downstream
vector or the match score, with stable input order as the final tie breaker.
The [conveyor pick example](../examples/conveyor_pick/conveyor_pick.json) supplies a
synthetic obstacle mask and runs with
`opencv-dataflow-run examples/conveyor_pick/conveyor_pick.json`. The check measures visible
image-plane clearance. Depth, gripper reachability and swept-volume collision
checks remain separate work.


## Available bead calipers

**Paired-edge bead calipers** samples parallel intensity scans and finds
opposing edges around a bright or dark bead. Set the scan endpoints across
the expected bead and the span along it. An optional pose makes the scans
follow the part. Width and center offset are measured in image pixels
without a pose and in local pose pixels with a pose. Edge coordinates are
always reported in the working image.

The block checks minimum edge gradient, width, center offset, passing-scan
fraction and the longest run of missing scans. A mask or region must cover
the full scan, including any interpolation neighbors that contribute.
Missing scans, including scans outside the image or selection, count
toward the gap. The [caliper example](../examples/bead_calipers/bead_calipers.json) measures
a 126-pixel synthetic strip and runs with
`opencv-dataflow-run examples/bead_calipers/bead_calipers.json`.
