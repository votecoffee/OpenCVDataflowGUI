# Learned same-grid image restoration

`ml.restore_image` runs one image-to-image ONNX head without resizing the source.
It returns the restored image, a processed-pixel mask, execution information and
optional ordinary DNN receipts. Use it for a compatible denoising/deblurring
export; the qualified pretrained baseline is the pinned grayscale DnCNN-25
denoiser. See [thirteen projects](<../examples/opencv tools/learned_restoration/README.md>),
nine with original arithmetic models and four with explicitly prepared weights.

## Models and sample values

Supply a local ONNX file, exact input name and at most one selected terminal
output. Empty output selection requires a model with one terminal head. Choose
input/output NCHW or NHWC independently. Native batch must be one and the output
must have exactly the input patch's height, width and working channel count.
Fixed-shape exports must accept every context shape used; there is no automatic
stretch, padding to a model size, output reshaping or scale guess.

Grayscale models require grayscale source images. RGB/BGR models accept tagged
RGB/BGR or RGBA/BGRA sources and return the source's original encoding. Alpha
passes through bit-for-bit. Use Convert image or Split/Merge channels when a
grayscale model is needed on color data. Physical depth, labels, mismatched color
tags and nonfinite/HDR float samples are rejected.

uint8/uint16 inputs are normalized by their integer full scale. float32 inputs
must already be in [0,1]. Working samples retain float32 precision; there is no
8-bit working quantization. Mean and Std supply one value or one per model
channel, in normalized source units. Input is `(unit - mean) / std`. Restored
output reverses that normalization; Noise residual output instead computes
`unit - prediction * std`. The model's actual head semantics must match the
selection. Blend mixes the reconstruction with original unit samples before
clipping to [0,1] and restoring source depth, with round-to-nearest for integers.
Information reports clipped samples and maximum unclipped change.

## Whole images, context tiles and selections

Whole processes the entire image once. Halo tiles divide it into disjoint
cores; each native call receives the core and Halo pixels of source context on
each side, clipped at the real image edges. Only the core is retained. Every
selected source pixel is written once; all output remains on the original grid.
There is no synthetic outer padding or interpolation/blending between tiles.
The native model retains its own internal padding behavior.

A sufficiently large halo reproduces whole-image processing only for a
compatible local model. Global attention, spatial statistics, positional inputs,
downsampling phases or other nonlocal behavior can prevent equivalence. The
caller must qualify the model, core dimensions and halo against whole-image
outputs. The pinned 17-convolution DnCNN has a 17-pixel receptive-field radius;
its retained whole/tiled discrepancy is below 2e-6 normalized units on these
controls. A zero-halo control differs by up to 30 integer samples at tile seams.

Mask and Region intersect as source-grid write selections. Excluded pixels
still supply model context, and outside the selection remains bit-identical to
the source. Unselected cores skip inference. Empty selection or Blend zero
returns the current original image with zero processed pixels and no model I/O.
Connected selection inputs are required when scheduled; known frame conflicts
are rejected. Same-grid output preserves source spatial/pyramid transforms,
tickets, recipes and arbitrary payload. Processed mask means pixels written,
not independent measured support or guaranteed defect preservation.

## Identity, execution and limits

Every active invocation freezes bounded model bytes once, including raw paths.
All tiles use that snapshot even if the source model file changes mid-run.
The next activation verifies current bytes again. Sealed contract bundles also
check artifact identity, block settings and preprocessing; Save As copies their
model/contract bytes. Raw model paths remain ordinary external references.
Changing sealed parameters requires a new reviewed bundle, or explicit raw-model
experimentation. `export-model` and the Python model-contract API support this
block; pass its `preprocessing(parameters)` declaration when sealing expected
preparation. Configuration is sealed independently of that optional declaration.

Inference reuses the shared DNN loader, backend/target/fusion controls, explicit
fallback policy and summary/detailed/off receipts. Classic preserves the selected
classic-engine import path; CPU avoids unnecessary accelerator discovery.
Receipts add the original core/context bounds per call. Available/requested
placement is not measured per-layer execution. Each block owns its cached native
network; independent blocks can execute in parallel. Native/shape failures reset
the cache and emit no new result event; a valid later invocation can recover.
Common timing, ticket judgments, logs and failed-input policies stay with the
base executor. Successful restoration means valid execution, not fidelity approval.

Preflight limits cover source pixels, planned native calls, combined context
pixels including repeated halo work, and expected same-grid output elements per
call. Model snapshots have the common 256 MiB guard. Each output is checked for
finite values and exact layout/shape. These guards do not bound native workspace,
all unretained terminal outputs or total process memory.

## Qualification

Independent filtering references verify arithmetic heads, direct/residual units,
NCHW/NHWC, all supported source encodings/depths, selection and unchanged alpha.
The real DnCNN export uses unchanged pinned MIT source weights, a deterministic
ONNX operator graph and direct-source tensor parity (maximum observed error
2.2352e-6 normalized units). PyTorch/ONNX are optional preparation/reference tools;
application inference uses OpenCV. See [provenance](<../examples/opencv tools/learned_restoration/SOURCES.md>).

The retained known-image benchmark compares Gaussian, median and learned outputs
with clean pixels, thin cracks and fine texture. DnCNN reduces noisy-control MSE
from 628.3195 to 14.4515, but erases all 29 faint-crack pixels under its stated
threshold. On low-noise input, MSE worsens from 4.1516 to 11.8902. The strong
crack survives. These are controlled outcomes for this export, sampling and
thresholds, not general accuracy claims or evidence of physical inspection
acceptance. The examples retain both improvement and failure cases.
