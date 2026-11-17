# SimpleFringeScan initial result reviews

These native OpenCVDataflowGUI projects load retained maps and, only for the two
known-calibration analytic controls, accepted XYZ with colored **fly-around cloud
thumbnails**. Open a project and run its file sources. No plugin, environment
installation or hardware is needed for review. File-source PASS means loading
only; no phase decoding or reconstruction executes in these saved reviews.

| Project | Available result | Scope |
| --- | --- | --- |
| [Analytic clean](studies/analytic_clean/project.json) | 6,912/6,912 metric samples | Independent rendered tilted-plane ray reference. |
| [Analytic missing](studies/analytic_missing/project.json) | 6,415/6,912 metric samples | Missing, clipped, dark and weak-phase regions remain gaps. |
| [External crown phase](studies/warsaw_crown_phase/project.json) | 56,985/559,872 phase pixels, 10.18% | CC BY4 external **simulated** images; phase-only, no metric cloud. |

Select Overview to see source, cyclic phase, amplitude, sinusoidal fit residual,
availability and depth where calibrated. Named source-aligned maps support zoom.
Only accepted finite XYZ rows become cloud points; raw maps preserve rejected
locations and ray/fit evidence. Camera/projector calibration is always explicit.
Neither wrapped-phase values nor missing pixels are converted into fake height.

All examples use identical generic decode/phase/geometry settings. The crown has
many low-signal/poor-fit regions; those remain visible under the defaults. Its
primary data and exact transformation are retained with
[dataset attribution](DATASET_CREDITS.md): **Michał Własiuk / Robert Sitnik**, Warsaw
University of Technology, *Simulated structured light 3D scanner datasets*,
[Zenodo](https://doi.org/10.5281/zenodo.17826191), **CC BY 4.0**.

The controls' maximum rendering-reference XYZ discrepancy is about 1.04e-8 m. This
checks the declared mathematical construction; it is not physical accuracy.
The external crown's standard-PBRT sRGB-inverse assumption and 3×3 linear-area
averaging are recorded explicitly, with original RGB/command PNGs and hashes.

## Replay the estimator

The standalone package has a CLI. After installation, from this examples folder:

```powershell
simple-fringe-scan studies/analytic_missing/capture.json --output new-metric-review
simple-fringe-scan studies/warsaw_crown_phase/capture.json --output new-phase-review
```

Published examples also include a wheel and `run_capture.py` to explicitly select
that wheel without installing into the host. Use the Python that runs the host:

```powershell
python run_capture.py --wheel releases/opencv_simple_fringe_scan-0.1.0.dev1-py3-none-any.whl studies/analytic_clean/capture.json --output new-review
```

Use a new output directory. `--no-graphics` preserves every raw map without PNG
generation. `maps.npz` keeps source coordinates, NaN gaps and discrete reasons;
`report.json` separates decoded from metric availability. A metric result exports
`accepted.ply` in meters with the named physical camera frame. Phase-only exports
have no PLY, metric XYZ or calibrated depth.

## Rebuild

`fetch_warsaw.py` downloads only 15 pinned members using bounded HTTP ranges and
verifies member SHA-256 plus ZIP CRC. It checks the primary license and publisher
archive identity before fetching; no whole-archive checksum claim is made.
`build_examples.py` rebuilds controls and the external derived phase example into
a fresh folder. `build_review_projects.py` adds/reopens the native saved reviews
through the host API. Processing/review never downloads data implicitly.

The [E2 setup and execution guide](README.md) adds retained reference and
known-calibration teaching, independent previews, Stage/Apply and a cached host
execution plugin. These original file-only studies remain separate. Acquired
physical captures, independent geometry accuracy, projector response calibration,
motion/coding-error checks and scalable stream processing remain open.
Phase fit/coverage is not defect acceptance or measurement accuracy.
