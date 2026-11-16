# Relative frequency response

Open one of these projects with **File → Open…**, then **Run once**. Image keeps
the original sample. Gain/Phase are frequency maps; Supported/Phase valid identify
measurable bins. Aperture shows the weighted image selection. Metrics contains
frequency coordinates and radial summaries; the result graphics include a radial
gain plot. Dark invalid map bins are placeholders, rather than measured zeros.

| Project | Expected result |
| --- | --- |
| [identity.json](identity.json) | Pass; identical uint16 captures have unit gain and zero phase. |
| [binomial_blur.json](binomial_blur.json) | Pass; periodic 3×3 binomial blur attenuates known frequencies. |
| [shift_phase_fail.json](shift_phase_fail.json) | Deliberate fail; a (-3,+2)-pixel shift keeps unit gain but violates the five-degree phase limit. |
| [gain_half_fail.json](gain_half_fail.json) | Deliberate fail; exactly half brightness violates the 0.8–1.2 mean-gain range. |
| [horizontal_grating.json](horizontal_grating.json) | Pass; only one conjugate frequency pair is excited, with unit response. |
| [masked_identity.json](masked_identity.json) | Pass; a circular Hann aperture excludes the changed exterior. |
| [color_red.json](color_red.json) | Pass; semantic Red has exactly half gain in BGR images. |
| [no_reference_energy.json](no_reference_energy.json) | Deliberate fail; a constant centered reference cannot support a response estimate. Scalar measurements emit nothing. |
| [budget_guard.json](budget_guard.json) | Deliberate validation error; the 1 MiB estimate rejects the transform before FFT. |
| [grid_guard.json](grid_guard.json) | Deliberate validation error; independent file-source frames do not satisfy strict Same frame. |

Reference samples are multiples of 16. Circular `[1,2,1]/4` filtering on both axes
therefore remains exact uint16 without quantization, with analytic transfer:

```text
H(fx, fy) = (1 + cos(2*pi*fx)) * (1 + cos(2*pi*fy)) / 4
```

The shift has known phase `-2*pi*(-3*fx + 2*fy)`, wrapped into [-pi,pi]. The fixtures
avoid using an estimated registration as their own truth. Tests compare maps to
these formulas, preserve source depth/tickets, and reproduce all project/PNG bytes
in Debug and Lean. Same frame checks metadata, not scene alignment; Pixel grid
asserts alignment without fitting it.

All nine PNGs, generators and projects are original work by OpenCVDataflowGUI
contributors under the repository's [MIT license](../../../LICENSE). They contain
no customer/third-party images. On-canvas comments carry attribution and scope;
`truth.json` records asset hashes and expected states. Rebuild from the repo root:

```powershell
.venv\Scripts\python.exe "examples/opencv tools/frequency_response/build_example.py"
```

See [the block contract](../../../docs/frequency-response.md) for floors, phase
validity, region/window effects and resource limits. These synthetic periodic
tests do not establish optical MTF certification from real camera images.
