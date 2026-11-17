# SimpleDeflectometry initial result reviews

Open a saved project below in OpenCVDataflowGUI and run its image sources.
No plugin, installation or hardware is needed for these **file-only reviews**.
PASS means result images loaded; these graphs do not run the estimator.

| Project | Numerical estimator result | Evidence |
| --- | --- | --- |
| [Analytic clean](studies/analytic_clean/project.json) | PASS; 6,912 comparable pixels | Independent unchanged phase control. |
| [Analytic change](studies/analytic_change/project.json) | FAIL at configured phase-change limit | Known localized phase change, no metric shape truth. |
| [Analytic missing](studies/analytic_missing/project.json) | FAIL; 6,212/6,912 comparable, 89.87% | Missing and clipped regions remain unavailable. |
| [Real SOFAST self-review](studies/sofast_self_review/project.json) | Diagnostic self-comparison; 114,356/114,359 comparable | Actual camera images, unknown exposure/response, no held-out defect or height truth. |

Each project has 16 source-aligned RGB review images: reference/test source,
X/Y phase, signed changes, magnitude, modulation, fit, quality, overlay and
overview. Shared fixed scales make comparisons useful; raw numeric arrays stay
in `result/maps.npz`, alongside the exact sequence/radiometry/decision report.
Purple means unavailable in phase/change views. The quality legend distinguishes
reference/test/both unavailable and outside ROI. Overlay red marks a configured
phase-change exceedance; availability colors remain separate.

All captures use the same generic phase-fit settings, 95% comparable-coverage
gate and illustrative 0.45-radian phase-vector limit. The SOFAST self-case has
zero change by construction; its useful result is code-level fit/coverage.
Wrapped differences remain whole-period ambiguous. There is no metric cloud or
phase-derived height image. The separate known-plane normal API has mathematical
controls; it does not reconstruct a measured shape from these phase images.

Real-data attribution: **OpenCSP Team**, Sandia National Laboratories NSTTF,
[OpenCSP](https://opencsp.sandia.gov/), **CC BY-SA 4.0**. Retained originals,
exact license, hashes, mask extraction, code-value reduction and limitations
are documented in [DATASET_CREDITS.md](DATASET_CREDITS.md). These SOFAST assets
and derived results retain their separate data license.

## Execute a fresh comparison

From the source repository, install the local FringeScan and Deflectometry
packages as described by the tool README, then run `simple-deflectometry` with
the reference and test capture manifests. The published host examples also
include both wheels and an explicit no-install launcher. From this folder,
using the host's Python:

```powershell
python run_capture.py --fringe-wheel releases/opencv_simple_fringe_scan-0.1.0.dev2-py3-none-any.whl --wheel releases/opencv_simple_deflectometry-0.1.0.dev1-py3-none-any.whl studies/reference/capture.json studies/analytic_change/capture.json --maximum-change-rad 0.45 --output new-change-review
```

Use a new output folder. `--no-graphics` preserves maps/report without PNGs.
`verify_package.py` selects both trusted wheels and replays all four cases;
`build_review_projects.py --verify-only studies` reopens native file-only views.
Preparation/review never downloads data implicitly. `prepare_sofast.py` accepts
the pinned official ZIP/license and requires h5py only for numeric mask extraction.

Native retained reference/region teaching, independent popup previews, Apply/Cancel
and the host execution block are next. Acquired calibrated reference/test pairs,
response calibration, absolute screen decoding and physical qualification remain
open. See [the application scope](APPLICATION_PLAN.md) and [validation](E1_VALIDATION.md).
