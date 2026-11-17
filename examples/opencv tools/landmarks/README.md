# Ordered landmarks and heatmaps

Open any project below and run once. The normal application environment includes
OpenCV DNN; no weights, downloads or extra runtime are needed. The original
120-by-80 uint16 BGR images contain three independently colored Gaussian spots.
A tiny ONNX model exposes their image-channel heatmaps and calculates intensity
centroids. Moving a spot moves the inferred result; there are no learned weights.

| Project | What to inspect |
| --- | --- |
| [refined.json](refined.json) | Log-quadratic fitting recovers three subpixel source centers. |
| [channels_last.json](channels_last.json) | The same heatmaps exported as NHWC. |
| [integer.json](integer.json) | Integer peaks show the effect of disabling refinement. |
| [letterbox.json](letterbox.json) | A wider source uses rounded letterbox preprocessing and padding exclusion. |
| [coordinates.json](coordinates.json) | Explicit X/Y/confidence rows from intensity-centroid arithmetic. |
| [normalized_coordinates.json](normalized_coordinates.json) | Those rows normalized by the full network dimensions. |
| [subset.json](subset.json) | Explicit green/red channel order with stable logical indices. |
| [selection.json](selection.json) | A source-grid rectangle limits the eligible red-channel peak. |
| [pose_region.json](pose_region.json) | Two ordered landmarks define a pose and local inspection rectangle. |
| [partial.json](partial.json) | A weak blue spot intentionally fails; its record stays in place and compact Points is suppressed. |
| [empty.json](empty.json) | A black image intentionally fails; tied zero maps remain diagnostic rejected records. |
| [pinned.json](pinned.json) | A sealed model/settings bundle and execution receipt qualify portable inference identity. |

The graphics overlay feeds the final-filmstrip-image override. Debug displays
green/red landmark markers, confidence labels and selected links; the pose example
adds a local region. Lean skips graphics while keeping the numerical results and
judgments. The original image output keeps its 16-bit pixel values and encoding.
Partial/empty are expected **inspection failures**, not program errors.

`truth.json` records expected ordered identities, source positions, judgments and
seven asset hashes. Gaussian fixtures have analytic centers; direct coordinate
fixtures use independently calculated finite-image intensity centroids, which
can differ from a Gaussian peak. Tests allow 0.002 source pixels for ordinary
controls and 0.06 for the bilinearly sampled wide letterbox image. These are
controlled numerical tolerances, not model or camera accuracy claims.

The ONNX outputs use input-dependent identity color channels plus sums, products,
maximums and divisions. Explicit tiling avoids relying on ambiguous importer
broadcasting. The sealed bundle includes the same model bytes and full inference
configuration. Decoder alignment/fitting settings remain saved block parameters.
See the [landmark contract](../../../docs/landmarks.md) before adapting another
model's axes, normalized coordinates or pixel alignment.

`build_example.py` regenerates into an empty destination; existing assets are
preserved by refusing to overwrite them. Automated checks build in a temporary
folder and compare assets and projects without modifying this collection.

All images, ONNX arithmetic, graphs and numerical controls in this folder are
original work by OpenCVDataflowGUI contributors, under the repository's
[MIT license](../../../LICENSE). The attribution is also present on every canvas.
No customer images or pretrained assets are included.
