# Learned image restoration

Open a saved project and Run once. Inspect Restore image with ONNX for current
pixels, processed selection and native-call information; click the error blocks
to compare with the separate, known aligned clean reference. All source images
are original synthetic tool controls. They are not production accuracy evidence.

| Project | Demonstration |
| --- | --- |
| `whole.json` | Original averaging model with an independent filtering reference; no learned weights. |
| `tiles.json` | Same arithmetic result using uneven disjoint cores and one-pixel context. |
| `residual.json` | A noise-residual output with explicit reconstruction semantics. |
| `rgb_alpha.json` | RGB model interpretation, original BGRA output and unchanged alpha. |
| `gray16.json` | uint16 source/output with float32 work and no 8-bit quantization. |
| `nhwc.json` | Explicit NHWC input/output model layout. |
| `selected.json` | Region writes only; excluded pixels remain context and retain exact source samples. |
| `empty.json` | Off-image region returns the current source without model I/O. |
| `fidelity_alarm.json` | Valid arithmetic inference with an intentionally failed zero-error quality check. |
| `native_whole.json` | Optional pinned DnCNN-25 denoiser, whole-image inference. |
| `native_tiles.json` | Same learned model with 17-pixel context; compare to a whole-image branch. |
| `native_no_halo.json` | Intentionally inadequate context; the seam-error check fails. |
| `native_low_noise.json` | DnCNN-25 on low-noise input; its MSE exceeds the five-sample-squared limit. |

The first nine projects need no downloads. The last four require explicit
preparation from the repository root:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/learned_restoration/prepare_model.py"
```

Preparation needs optional `torch` and `onnx` authoring packages. It downloads
the exact MIT Kai Zhang DnCNN-25 weights, verifies their size/hash before
weights-only CPU loading, and creates deterministic sealed ONNX bundles in
ignored `.local` folders. It does not train, change weights, execute downloaded
Python, or install dependencies. For offline source weights append
`--weights path/to/dncnn_25.pth`. Repeated preparation validates matching existing
bundles without network use and preserves/rejects different content. The app
itself executes prepared models with OpenCV and needs neither authoring package.

The learned model lowers noisy-control MSE from 628.3195 to 14.4515, but removes
all 29 faint-crack pixels under the recorded threshold. On low-noise input MSE
gets worse, from 4.1516 to 11.8902. With 17-pixel context, tiled/whole difference
stays below 2e-6 normalized units on these controls; zero context creates seams
of up to 30 integer samples. Valid inference therefore does not imply acceptable
inspection results. No physical dimensional accuracy is established.

`truth.json` contains exact asset/contract identities, feature definitions and
ROIs. `benchmark.json` retains direct-native learned/filter comparisons,
false/erased crack counts and flat/texture errors against clean known pixels.
`benchmark.py` reruns them independently; optional `--weights` also checks source
tensor arithmetic. `build_example.py` regenerates original assets/projects and
preserves differing assets. Model-bound settings are deliberately sealed; export
a new bundle for changed settings or explicitly use a raw model while developing.
The static clean reference has its own source identity; comparisons here use the
known corresponding pixel grids, not live-camera registration or synchronization.

Tests: `tests/test_learned_restoration.py` and
`tests/test_learned_restoration_examples.py`. See the
[full block contract](../../../docs/learned-restoration.md),
[sources and conversion](SOURCES.md), and retained [MIT model notice](MODEL-LICENSE.txt).
Every project includes an attribution comment; no customer images are included.
