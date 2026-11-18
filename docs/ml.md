# Image models, LightGlue and Annoy

**OpenCV > Deep Neural Networks (dnn)** contains DNN inference, classification
and embedding tools. **Custom > ML** includes LightGlue matching and Annoy
vector search. ML borders use pink. Image, pose, region and
data connections keep the ordinary canvas type colors. All these blocks use the
existing ticket/outcome, payload, recipe-parameter, timing and diagnostic contracts.
The index writer is under **Custom > Output > ML models > Save Annoy index**.

Reusable native color palettes and image-descriptor vocabularies use the base
OpenCV installation. Train, save, reload and predict without refitting each
capture; distance/cluster filters report unknown samples. See
[reusable clusters](reusable-clusters.md) for preprocessing and model contracts.

Native [K-nearest image classification](knn-classification.md) adds portable
training/reload, neighbor votes/distances and explicit unknown-image rejection.
It shares descriptor validation and normalization with SVM, preserving saved
preprocessing contracts. Both run with the base OpenCV installation.

Native [random forest classification](rtrees-classification.md) adds explicit
tree/split controls, class priors, sample weights, optional feature importance
and per-class votes. Portable models retain checked native trees and their
training/preprocessing settings. This also uses the base OpenCV installation.

Native [ALIKED/LightGlue ONNX blocks](native-lightglue.md) run through OpenCV 5
without PyTorch and live under **OpenCV > Features Framework**. Existing Python
LightGlue workflows below retain their settings and optional dependencies.

Native [image embedding retrieval](native-embedding-index.md) also uses the base
OpenCV installation: build/search/save/load galleries with stable image IDs,
checked feature contracts and self-contained native bundles. HOG and binary
image hashes connect directly; approximate recall remains explicit.

Opt-in [DNN profiling variants](dnn-profiling.md) measure native layers on the
same single-image, multi-input or batch forward. CPU timing availability, fusion
controls, per-call source identity and an independent off switch remain explicit.

[Soft-NMS](soft-nms.md) adds linear/Gaussian score decay on existing detection
collections, with a hard/disabled comparison, explicit native integer-box rounding,
class grouping and aligned source indices/scores. Original geometry is preserved.

## Installation

Point/box [prompted segmentation](prompted-segmentation.md) uses the pinned
EfficientSAM model with separate prompt, inference and mask-decoding blocks.
Three native candidates, raw predicted qualities and exact source/request binding
remain inspectable. [Eight examples](<../examples/opencv tools/prompted_segmentation/README.md>)
include an openly licensed photograph and independent exact shape truth.

OpenCV DNN is included in the base OpenCV 5 contrib installation. Bundled control
models need no additional packages or downloads. The optional pretrained detector
uses its explicit model-preparation script. For Annoy:

```powershell
.\.venv\Scripts\python -m pip install -e ".[annoy]"
```

Annoy is a compiled extension. If pip cannot find a compatible wheel, its source
build needs a C++ compiler; Windows development here successfully built Annoy 1.17.3.

For LightGlue, first choose a CPU or CUDA PyTorch build appropriate to the machine
using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/).
The following installs the CPU variant used for local verification, then the
remaining runtime dependencies and the pinned official LightGlue source:

```powershell
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e ".[lightglue,ml-test]"
.\.venv\Scripts\python -m pip install --no-deps "lightglue @ git+https://github.com/cvg/LightGlue.git@5a650c444718caffc2669dc0094cde908f42dd73"
.\.venv\Scripts\python scripts/prepare_ml_assets.py --lightglue
```

Install official LightGlue with `--no-deps`: its upstream requirements name
`opencv-python`, which would coexist incorrectly with this project's required
`opencv-contrib-python`. The extra explicitly installs the other dependencies;
contrib supplies `cv2`. Dependency checkers may still report the upstream
`opencv-python` metadata requirement. Do not install a second OpenCV distribution
to silence that metadata mismatch.

The preparation script explicitly downloads **SIFT LightGlue** weights to ignored
`.local/models/sift_lightglue.pth`, records their URL/size/SHA-256, and preserves
existing weight files. It also generates a tiny local RGB-average ONNX network.
`onnx`, supplied by `[ml-test]`, is only needed to regenerate that synthetic model.
The included `examples/assets/rgb_average.onnx` is generated from this script, has
no pretrained parameters, and is intended to verify wiring and preprocessing.

Graph execution uses local paths and never downloads weights. Relative model paths
resolve beside the project. Keep application models in ignored `.local/` or another
chosen artifact directory; store paths in recipe parameters when selecting models
by recipe. Native networks/indexes are cached per execution block and released on
close/reset, rather than serialized into saved projects or copied between workers.
Save As rebases DNN model input references while leaving those external files intact.

## OpenCV DNN

**DNN ONNX inference** accepts an image and optional region/mask. Selected-out
pixels are zeroed before resizing. Configure input size, RGB/BGR/grayscale encoding,
stretch or letterbox resizing, padding, per-channel mean/std, scale, input name,
output names, engine/backend/target and fusion. Enums use inspector dropdowns.
The normalization is `(pixel - mean) * scale / std`, applied in the chosen channel
order. Pixel values preserve their source units: uint8 0..255, uint16 0..65535,
or the original floating-point range. Gray input requires one mean/std value;
RGB/BGR requires three. The default RGB scale of 1/255 is intended for uint8.

The engine dropdown exposes OpenCV 5 Auto, Classic, New and ORT engine selectors;
availability depends on the installed wheel. The initial default is CPU. Non-CPU
targets and disabling fusion explicitly require Classic. Unsupported targets fail
with a diagnostic rather than silently claiming GPU execution. CPU inference was
checked against Auto, Classic and New using the real included ONNX model. CUDA,
OpenCL and ORT support are not claimed as locally validated.

The `tensors` output maps layer names to JSON-compatible `tensor_v1` records with
shape, dtype and flattened numeric values. `output_names=[]` selects all terminal
outputs. `max_output_elements` bounds total retained output values; this is an
output guard, not a limit on the network's internal allocations. `preprocessing`
contains exact input-to-network/network-to-input transforms, including rounded
letterbox dimensions and padding, for model-specific postprocessing.

**Classify DNN tensor** selects an output, applies optional stable softmax, assigns
labels and returns top-K predictions. Confidence/accepted-label criteria produce a
normal failed inspection and a `classification` reason when rejected. Softmax-off
requires probability values in [0,1]. This decoder is for single-image class scores.
**Decode object detections** accepts explicit YOLO-style XYWH score/objectness,
postprocessed XYXY and SSD row layouts. It expands declared YOLOX grids when
requested, filters classes/scores/selected centers, applies class-aware or
class-agnostic native NMS and maps boxes back through preprocessing. It emits
sortable/filterable source-frame collections and optional drawable labels/boxes.
The generic inference block does not guess an export schema. See
[detection contracts](object-detection.md) and [control/real-photo projects](<../examples/opencv tools/dnn_detection/README.md>).
Instance mask decoding is covered below.

**Decode rotated detections** handles explicit oriented score/objectness/class
heads, angle units/directions, corrected or native rotated suppression and exact
source polygons/affine poses. Sort/filter collections and retain their supplied
poses for local inspection regions. See [contracts](rotated-detection.md) and
[twelve controls](<../examples/opencv tools/rotated_detection/README.md>).

**Decode semantic segmentation** accepts exclusive multiclass scores, binary
foreground scores or decoded class IDs. It restores full-field heads through
exact preprocessing maps to source-grid labels, selected masks, confidence and
class/area statistics. Choose nearest labels or linear raw scores; ignored IDs,
ROI selection, confidence thresholds and area judgments remain explicit. Use the
mask with ordinary blob/contour/morphology tools. See [contracts](semantic-segmentation.md)
and [twelve controls](<../examples/opencv tools/semantic_segmentation/README.md>).

**Decode instance segmentation** adds box-relative/class masks and coefficient/
prototype masks, retaining separate source-grid regions through NMS and overlap.
Pack separate boxes/scores/classes when needed; sort/filter by mask metrics and
select an exact individual mask for further inspection. See
[contracts](instance-segmentation.md) and
[twelve controls](<../examples/opencv tools/instance_segmentation/README.md>).

**Multi-input ONNX inference** now binds up to four explicitly named float32
tensors. **Image to DNN tensor** prepares each image independently; **Create
float32 tensor** adapts numeric data or constants. Declared shapes, complete
bindings, Current/Reference roles and ticket policies remain explicit. Primary
input owns result coordinates and optional preprocessing. Shared recipe rules
still apply. See [contracts](multi-input-inference.md) and
[nine numerical examples](<../examples/opencv tools/multi_input_inference/README.md>).
All three inference interfaces support pinned model/settings bundles and
Summary/Detailed/Off execution receipts. Byte/settings mismatches reject;
opt-in unavailable-target CPU fallback is reported. `export-model` publishes a
new bundle without processing images. See [contracts](model-contracts.md) and
[ten controls](<../examples/opencv tools/model_contracts/README.md>).

**Batched ONNX image inference** accepts ordered image groups, chunks them into
native NCHW calls and preserves a result row for each real image. Declared shapes,
dynamic source sizes, variable/padded/rejected tails, per-member selections and
resource limits are explicit. **Unpack image inference result** exposes each
row's original image, tensors and preprocessing to existing decoders after
Select group entry or Unbatch. See [contracts](batch-inference.md) and
[ten numerical projects](<../examples/opencv tools/batch_inference/README.md>).

The inference blocks fetch all native terminal outputs once before selecting a
requested subset/order, accommodating this OpenCV 5 engine's complete-output
requirement. Unselected output allocations remain part of native network work.

**Tensor to embedding vector** selects a tensor and flattens it or globally averages
all axes except a configured channel axis. It accepts one-image batches, with
optional L2 normalization. A zero vector cannot be L2 normalized.

Try [DNN embedding](<../examples/opencv tools/dnn_embeddings/dnn_embeddings.json>), or:

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/opencv tools/dnn_embeddings/dnn_embeddings.json" --repeat 3
```

The synthetic network averages RGB channels. Its labels are channel names, not
semantic object recognition. Replace its ONNX path and preprocessing with those
required by a real application model.

**Decode keypoints and heatmaps** recovers ordered single-object landmarks from
explicit heatmap or X/Y/confidence heads, with source-grid resize/letterbox
mapping, optional mask/region, ambiguity policy and qualified local subpixel
fits. Rejected entries retain their indices; incomplete compact Points is
suppressed by default. **Pose from ordered landmarks** defines a local frame
from accepted points for pose-relative inspection. See [contracts](landmarks.md)
and [twelve original image-dependent controls](<../examples/opencv tools/landmarks/README.md>).

## LightGlue reference matching

The original adapter uses OpenCV SIFT/RootSIFT descriptors with the matching
learned SIFT checkpoint. A [separate external SuperPoint feature path](local-features.md)
packs declared 256-value descriptors and uses SuperPoint-trained LightGlue weights.
Its real detector/extractor qualification remains open; SIFT/ORB records cannot
substitute for that descriptor family. [DISK extraction and matching](disk-features.md)
now run with local depth/epipolar detector weights and the matching LightGlue
checkpoint. Use `prepare_ml_assets.py --disk` to prepare all three pinned files.
[ALIKED extraction and matching](aliked-features.md) now support the n16,
n16rot and n32 presets, native subpixel/NMS controls and an independent strict
score floor. Kornia 0.9 or newer is required; `prepare_ml_assets.py --aliked`
prepares the three detector checkpoints and matching LightGlue weights.
The explicit `prepare_ml_assets.py --superpoint-matcher` option prepares only the
Apache-2.0 matcher, with a pinned SHA-256; no SuperPoint detector is downloaded.

**LightGlue SIFT features** captures a portable reference feature model containing
keypoints, RootSIFT descriptors, scales, orientations and coordinate context. It
accepts a mask/region, needs no learned weights, and exposes a `train(inputs)` API.
Connect its `features` output to **LightGlue feature matcher**'s
`reference_features` input to reuse a captured reference. The matcher can instead
extract a supplied `reference` image, with its own optional reference mask/region.
Reference capture does not fine-tune the learned neural network. These feature
records travel on data ports; they are not implicitly saved as matcher state.

The query accepts its own mask/region. Exposed controls include maximum features,
SIFT thresholds/octave layers/sigma, CPU/CUDA/Auto device, 1..9 learned layers,
FlashAttention option, mixed precision, adaptive depth/width confidence, match
filter threshold, robust homography method/threshold/confidence/iterations, and
minimum match/inlier count and inlier ratio. Mixed precision requires CUDA.
The local checkpoint is loaded with PyTorch's weights-only loader and checked for
the required SIFT matcher parameters; no random-weight inference is substituted.

Outputs include paired points and scores, geometric inliers, query overlay,
reference-to-query transform and an `image_pose_v1` pose. Pose origin corresponds
to reference pixel (0,0). Failed/degenerate matches emit no transform or pose event;
they retain match diagnostics and produce a `lightglue_match` fail reason.
Coordinates remain local to each input's image/pyramid level, with separate
reference/query coordinate contexts. Use the pose with existing pose-relative
region/inspection blocks. This is one planar registration, not repeated-part
multi-instance discovery.

Try [LightGlue matcher](<../examples/custom tools/lightglue_match/lightglue_match.json>) after preparing its weights, or:

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/custom tools/lightglue_match/lightglue_match.json"
```

The example captures a synthetic textured reference and locates it in a larger
scene. Native CPU tests also recover a known 8-by-6-pixel translation with the
official learned checkpoint and check that inference makes no download calls.

## Annoy vector search

**Vector data** supplies one vector or a row matrix for examples/static references.
**Build Annoy index** accepts a vector matrix and optional labels, with metric,
tree count, seed and normalization controls. Labels are strings; omitted labels
become sequential IDs. Its `annoy_index_v1` output is a portable model containing
training vectors and metadata, suitable for smaller reusable graph branches.
The build block verifies the native index once per changed model; search owns its
own native instance so handles are never transferred between execution threads.

**Annoy nearest neighbors** accepts a model and a vector (for example a DNN
embedding), with neighbor count, search effort (`search_k`), minimum result count
and optional filtering. All five Annoy metrics are exposed: angular, Euclidean,
Manhattan, Hamming and dot. Hamming requires binary vectors and no normalization;
angular requires nonzero vectors. Index and query use the same normalization.

Distance metrics return ascending `distance` values. Dot returns descending inner
product `similarity`, with `distance=null`; it must not be treated as a smaller-is-
better distance. `max_distance` filters distance metrics; `min_similarity` filters
dot, with -1 disabling that filter. Too few accepted neighbors is a normal failed
inspection (`nearest_neighbors`), not a native exception. Approximate search can
trade recall for latency through tree count and `search_k`.

For larger/reused indexes, **Save Annoy index** writes a new JSON metadata file and
`.ann` binary beside it. Existing files are preserved. **Load Annoy index** resolves
the binary relative to that metadata and outputs a file descriptor. Search verifies
the SHA-256 and item count before mapping it, caches the read-only native index,
and unloads it on close/reset. Optional prefault requests eager page loading.
Training data does not have to travel through the graph after loading a file index.
Opening a project does not load/map the index until it is run.

Try [Annoy search](<../examples/custom tools/annoy_search/annoy_search.json>), or:

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/custom tools/annoy_search/annoy_search.json" --repeat 3
```

This connects the synthetic DNN's RGB embedding to labeled reference vectors.

## Validation and upstream references

Normal tests use the included generated ONNX model. Annoy tests run when its optional
extension is installed; the learned LightGlue integration test also requires the
local checkpoint and optional runtime. These tests do not fetch packages or weights.

- [OpenCV 5 DNN engine selection](https://docs.opencv.org/doc/doxygen/html/de/db7/api_dnn_engine_selection.html)
- [Official LightGlue code and configuration](https://github.com/cvg/LightGlue)
- [Official Annoy API and metric behavior](https://github.com/spotify/annoy)

Upstream dependencies and pretrained assets retain their own licenses. The LightGlue
checkpoint is downloaded separately and is not redistributed in this repository.
