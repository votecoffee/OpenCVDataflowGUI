# ALIKED features and LightGlue

**Custom > ML > Feature matching** includes **ALIKED learned features**
(`ml.aliked_features`) and **LightGlue ALIKED feature matcher**
(`ml.lightglue_aliked`). They run real local-weight Kornia ALIKED extraction and
the matching learned LightGlue checkpoint. The optional ML extra now requires
Kornia 0.9 or newer; see [ML setup](ml.md).

Prepare the three detector presets and matching checkpoint explicitly:

```powershell
.\.venv\Scripts\python scripts/prepare_ml_assets.py --aliked
```

The script checks pinned SHA-256 values, records source/size/hash and preserves
existing files. Weights remain in ignored `.local/models`. Opening or validating
a project does not import ML runtimes, load weights or contact the network.
Execution uses the ordinary native constructor and weights-only local loading,
without a global download-loader patch. Every required learned parameter must
exist; only the deterministic, non-learned DKD radius grid may be absent from the
older upstream checkpoints.

## Extraction controls

**Model name** selects `aliked-n16`, `aliked-n16rot` or `aliked-n32`, paired with
its local checkpoint/digest. These have 128-value descriptors. The 64-descriptor
t16 model is outside this matched-LightGlue baseline. The architecture setting
does not authenticate training identity; the actual checkpoint hash identifies
the loaded weights.

**Detection mode** exposes two native policies:

- **Top K** retains up to **Max features** native local maxima. Detection threshold
  is unused in this mode.
- **Threshold** selects native peaks strictly above a positive **Detection
  threshold**, with the same cap. If none qualify, the verified Kornia backend
  falls back to the image's mean score. The later **Minimum score** is a separate,
  inclusive hard floor and can reject those fallback results.

**NMS radius** controls the positive native suppression/soft subpixel window.
Scores are sampled probabilities in 0..1, distinct from learned matching scores.
The qualified implementation is Kornia 0.9.0, including its strict-local-maximum
and per-image fallback behavior. It is not bit-for-bit parity with the original
ALIKED/LightGlue extraction ports or other backend versions.

The [shared learned-image handling](disk-features.md#extraction-and-reference-capture)
accepts uint8/uint16 or normalized float32 gray/color, correctly interpreting
tagged channel encodings and retaining the original Image output. Optional
downscaling exposes interpolation and maps source pixel centers with
`(network_xy + 0.5) / actual_scale - 0.5`. ALIKED internally replicate-pads around
the center to multiples of 32 and unpads its maps, so padding adds no output
offset. Source pixels are bounded before conversion; padded network pixels are
bounded before resampling or dense inference.

Mask, Region, alpha and Minimum score filter corresponding rows after native
detection/capping. They do not mask descriptor context or reduce CNN work.
`raw_indices` preserves identity in that native result. A connected selection
without an event prevents execution. `minimum_count` controls a normal judgment;
zero permits empty capture. Native/filtered counts, score policy, padding/scaling,
model preset and detector settings are reported.

`lightglue_aliked_v1` carries source-space points, 128-value unit descriptors,
probabilities, row IDs, image size, coordinate context and extractor digest/
backend/device/settings. Tickets, recipe, spatial/pyramid and processing-start
metadata survive. `train(inputs)` captures portable reference features without
fine-tuning weights or borrowing a previous invocation's ticket. Each instance
owns a lazy model cache; checkpoint/device or native detector-setting changes
rebuild it. Reset/close releases owned model references.

## Matching, diagnostics and scope

Connect Reference features, Query features and the current query Image. The
[shared matching contract](local-features.md#matching-and-inspection) enforces
descriptor family, source grid, known query/body/wire context and work limits;
reference acquisitions retain their own identity. SIFT, DISK and ALIKED records
are distinct despite sharing some descriptor dimensions.

The matched local LightGlue checkpoint exposes device, learned layers, attention,
mixed precision, adaptive depth/width and confidence controls. Robust homography
settings and match/inlier criteria govern one planar registration. A failed fit
or count emits no current pose or transform. The primary Image stays unchanged;
drawable inlier/outlier markers feed overlays and the filmstrip. Lean skips
optional painting. The base executor contains exceptions, records diagnostics/
timing and allows recovery. Failed-input admission preserves the last processed
image and performs no new native work.

[Thirteen projects](<../examples/custom tools/aliked_matching/README.md>) cover
all three presets, known source translations, resizing, RGB16, ROI/local poses,
native threshold fallback, a strict score floor, empty/count failures, genuine
SIFT comparison and separate attributed VisA PCB photographs. Public native API
parity, model completeness/cache/integrity, input gates, exception recovery,
JSON/XML/Save As, exact regeneration and Debug/Lean results are checked.

CPU qualification uses Python 3.14.3, OpenCV 5.0.0, Kornia 0.9.0 and
PyTorch 2.15.0+cpu. CUDA controls are exposed but accelerator behavior remains
unqualified. These controls establish a bounded planar image-pair baseline;
arbitrary exports, t16 matching, cross-version repeatability, repeated-instance
discovery and physical metrology/localization remain separate work.

References: [ALIKED source/paper](https://github.com/Shiaoming/ALIKED),
[Kornia local features](https://kornia.readthedocs.io/en/latest/feature.html),
[LightGlue](https://github.com/cvg/LightGlue). Attribution and model/image license
details are included with the examples.
