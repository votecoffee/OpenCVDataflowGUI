# OpenCV runtime and browser options

The project requires `opencv-contrib-python>=5.0.0.93,<6`. The local development
environment uses OpenCV **5.0.0**, Python distribution **5.0.0.93**. Its desktop UI is
PySide6/Qt, with native OpenCV Python bindings and NumPy image buffers. It does not
embed a browser or execute WebAssembly.

## Default processing and legacy compatibility

Use the ordinary **Image processing** blocks for development:

| Block | ID | Operation |
| --- | --- | --- |
| Grayscale | `cv.gray` | `cvtColor`, or grayscale passthrough; uint8/uint16/float32, encoding-aware, no implicit depth conversion |
| Gaussian blur | `cv.gaussian_blur` | `GaussianBlur`, independent X/Y kernels and sigmas, borders and constant border value |
| Canny edges | `cv.canny` | `Canny`, automatic uint8 grayscale working image, thresholds, aperture dropdown and L2-gradient option |

Gaussian blur and Canny accept masks/regions with the existing full-image
neighborhood policy. Blur preserves pixels outside selection; Canny zeroes them.
Grayscale converts the full image. Grayscale and Gaussian blur expose OpenCV 5's
algorithm-hint dropdown (default, accurate, approximate). Approximation is an
explicit choice; the default delegates to OpenCV's build policy. Each block retains
its own input/output images, metadata, result and timing for inspection.

The starter/demo, saved example projects and nested sample ToolBlocks now use
ordinary calls. Existing `gapi.gray`, `gapi.blur` and `gapi.canny` remain available
under **Advanced > G-API**, with unchanged IDs and parameters. Opening an older
user project keeps its chosen backend; it is not silently rewritten. To switch a
user project, replace those nodes with the ordinary equivalents and copy their
parameters/connections. Existing Gaussian and Canny parameter names match; ordinary
grayscale and blur additionally expose the algorithm hint. Ordinary Grayscale
supports additional input formats and depths; neither variant rescales
depth automatically. No project schema change is needed.

## Compiled ToolBlocks (future)

[Issue 4](https://github.com/votecoffee/OpenCVDataflowGUI/issues/4) tracks an
explicitly compiled ToolBlock for stable processing chains. This is planned scope,
not an implemented mode or a claim of improved speed. G-API's intended value is
optimized graph execution and backend portability, as described in
[OpenCV's Graph API documentation](https://docs.opencv.org/4.x/d0/d1e/gapi.html).
Using a node editor alone does not require compiling its graph.

The user would opt in on a compatible ToolBlock and select public result and
diagnostic outputs. Preserve the ordinary execution path for developing and
debugging all steps. Compile only an explicitly supported processing subset;
keep acquisition, buffered control flow, plugins, parallel inspection scheduling
and external writes outside the initial compiled region. Unsupported operations
must produce a clear explanation. Graph/parameter/type/diagnostic changes must
invalidate the compiled artifact. Preserve tickets, spatial/pyramid/custom payloads
and mask/region semantics, with numerical equivalence checks and documented
tolerances. Expose selected diagnostics and clearly state unavailable internal
results/timings; do not imply that every hidden intermediate is observable.

Before enabling this feature or expanding its subset, require a reproducible
measured benefit on the intended workflow:

1. Compare the identical chain, image sizes, selected diagnostics and outputs
   against ordinary execution using recorded representative inputs.
2. Record hardware, OS, OpenCV build/backend, thread settings and configuration.
   Measure compilation/cold start separately from warmed end-to-end median/p95
   latency, throughput and peak memory across repeated trials.
3. Include input/output copies, diagnostic materialization and application
   integration costs. Report output-equivalence checks alongside performance.
4. Require improvement beyond measurement noise that justifies added complexity
   and amortizes compilation over the expected job lifetime. If evidence shows no
   useful benefit, keep ordinary execution and defer expansion. Portability claims
   also require validation on each actual target backend.

## OpenCV 5 migration

The contrib wheel includes extra modules and retains the G-API functionality used
by existing grayscale/blur/Canny nodes. Their block IDs, parameters and saved
connections remain unchanged. GComputation now receives native argument lists
instead of the removed Python `GIn/GOut` helpers; no eager fallback is used.
Line-segment detection normalizes scalar-per-line output vectors before exposing
width, precision and NFA measurements.

The migration was checked against the complete automated suite, offscreen Qt
editor workflows and saved example graphs. Shape/ORB models trained and saved on
4.14.0 were also loaded and matched on 5.0.0 without retraining. This is synthetic
compatibility evidence; representative inspection images should determine actual
acceptance tolerances. OpenCV's [migration guide](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration)
documents array-layout, interpolation and rendering changes, so numerical results
are not promised to be bit-identical across major versions.

Install only one OpenCV distribution in an environment. The base, contrib and
headless packages all provide `cv2`; contrib replaces the base wheel. See the
[README upgrade commands](../README.md#install). The dependency exposes additional
native APIs to plugins; new contrib algorithms still need explicit block wrappers
to appear in the toolbox.

## Browser option

WebAssembly is worth considering for a future browser-based editor, viewer or
portable preview runner. It is not new to version 5:
[OpenCV.js builds](https://docs.opencv.org/4.x/d4/da1/tutorial_js_setup.html) already
target WebAssembly in OpenCV 4. A browser backend could process selected images
locally without a Python service, but would need an explicit supported-block
subset, graph/model interoperability and measurements of latency and image-copy
costs. Python plugins, device acquisition and filesystem/database writers would
need separate handling.

For the current native desktop runtime, there is no demonstrated benefit from
adding a WebAssembly execution layer. Keep this as a future backend option and
benchmark a concrete browser workflow before introducing it.
