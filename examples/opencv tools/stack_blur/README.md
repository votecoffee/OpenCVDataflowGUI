# Large-radius stack smoothing

Open a project and use **Run once**. **Stack blur** is under **OpenCV → Image
Processing (imgproc) → Filtering**. All fixtures are original MIT synthetic data.

| Project | What to inspect |
| --- | --- |
| [Stack versus Gaussian](stack_vs_gaussian.json) | Common float32 input, 127×127 support and matched kernel variance. Compare image results, column profiles and raw difference. |
| [Selected smoothing](selected_smoothing.json) | Smooth inside a circle while preserving all outside pixels. Result graphics outline the write selection. Neighborhoods still use the full image. |
| [16-bit smoothing](uint16_smoothing.json) | Float32 working arithmetic retains uint16 output; no automatic intensity normalization occurs in the filter. |
| [Unsafe native arithmetic](unsafe_native.json) | Intentional validation error: this build's native high-depth integer path is excluded. No output is emitted. |
| [Explicit kernel clamping](kernel_clamp.json) | A requested 255×255 kernel becomes 127×95 on the 128×96 input; Info shows both sizes. Choose Reject to make oversizing an error. |

The 1024×768 fixture has a step from 60 to 180 DN at continuous pixel coordinate
x=511.5, with deterministic quantized Gaussian noise. At kernel 127 the recorded
float32 measurements reduce flat-area RMS error from **18.044 DN** to **0.171 DN**
for stack blur and **0.172 DN** for Gaussian. The measured edge centers are
**511.569** and **511.573**. Their 10–90% edge widths grow to **70.71** and
**69.41 pixels**, respectively. Both smooth away edge sharpness; they are not
edge-preserving denoisers. This is software fixture evidence, not optical calibration.

[truth.json](truth.json) stores design and measured references. The Gaussian sigma
is 28.3070, chosen so its finite 127-tap kernel matches the triangular stack
kernel's variance of 682.5. Both methods use replicated borders and the same
support. This matches blur variance, rather than assuming equal kernel sizes
imply equal smoothing.

The separate [benchmark reference](benchmark_reference.json) records one local
1920×1080 grayscale float32 run with one OpenCV thread, two warm-ups and seven
alternating-order repeats. These are native-call medians, including native output
allocation and excluding graph copies, validation, conversion and graphics:

| Kernel | Stack (ms) | Gaussian (ms) |
| --- | ---: | ---: |
| 31×31 | 9.490 | 9.455 |
| 127×127 | 10.149 | 42.726 |
| 255×255 | 10.804 | 106.137 |

These observations depend on hardware, backend, image layout and system load.
They do not establish a guaranteed GUI speedup. Repeat the benchmark in a fresh
process and supply a new report path:

```powershell
python "examples/opencv tools/stack_blur/compare_filters.py" --report new-benchmark.json
```

Rebuild deterministic fixtures, references and projects:

```powershell
python "examples/opencv tools/stack_blur/build_example.py"
```

See [the block contract](../../../docs/stack-blur.md) for depth, alpha, selection,
work limits and native integer qualifications. Images, builder and benchmark
code are original work under the repository MIT license. Each project also
contains an attribution/comment block.
