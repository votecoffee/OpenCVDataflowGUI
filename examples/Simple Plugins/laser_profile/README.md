# SimpleLaserProfile saved-result reviews

Open one of the projects below and run it. Source image, stripe overlay and
profile chart use standard image outputs. The calibrated profile uses the
native **fly-around point-cloud thumbnail and popout**, with meters, camera
frame and RGB colors. Double-click the cloud and enable Flyaround to orbit it.

These projects review retained results; they **do not run stripe extraction**.
File-source PASS means loading succeeded, not inspection or gauge acceptance.

| Project | What to assess |
| --- | --- |
| [Generated 50-mm step](studies/calibrated_control/step/project.json) | 49.998-mm result; 480 measured points. Generated software control only. |
| [Missing/competing/clipped-intensity control](studies/calibrated_control/unavailable/project.json) | 90 rejected bins stay unavailable; 390 accepted cloud points. |
| [Retained teaching reference](studies/calibrated_control/reference/project.json) | Original source and calibrated reference output. |
| [Real WLI reference](studies/wli/capture_1/project.json) | Fixed taught ROI; 72/120 accepted bins, 48 at input ceiling. |
| [Real WLI second capture](studies/wli/capture_2/project.json) | Same recipe; 42/120 accepted, missing and competing evidence visible. |
| [Real WLI poor fit 3](studies/wli/capture_3/project.json) | 2/120 accepted; ROI clipping and poor reference agreement retained. |
| [Real WLI poor fit 4](studies/wli/capture_4/project.json) | 7/120 accepted; poor fit retained. |
| [Unrestricted multi-stripe diagnostic](studies/wli/unrestricted_diagnostic/project.json) | All 800 bins ambiguous/overflowing; outside taught scope. |

Green marks accepted centers, magenta competing peaks, red the input encoding
ceiling, amber clipping/width failures, cyan the ROI, and blue missing bins at
the image border. The plot retains gaps and has an availability strip. Reports
keep every source bin and rejection reason. Coverage is not an accuracy score.

WLI-Set v3 is real **CC BY 4.0** data by **zeng song**, credited in
[DATASET_CREDITS.md](DATASET_CREDITS.md), with byte-exact originals and published
centerline masks. Those captures have no camera/laser-plane calibration and
remain pixel-only, without metric clouds. Nearest software-centerline distance
does not establish identity, completeness or physical accuracy.

## Optional standalone teaching and execution

The included `opencv-simple-laser-profile` dev1 wheel provides the NumPy/OpenCV
API and CLI, retaining reference, source ROI, background and calibration in a
portable `.slp` recipe. Use a separate environment or an explicitly configured
tool environment. No installation is needed to open these review projects.

```powershell
python -m pip install releases/opencv_simple_laser_profile-0.1.0.dev1-py3-none-any.whl
simple-laser-profile inspect studies/calibrated_control/trained.slp
simple-laser-profile run studies/calibrated_control/trained.slp studies/calibrated_control/step.png --output new-step-review
simple-laser-profile train "studies/wli/origin/s_multi_10 (1000).jpg" --roi 280 235 120 50 --recipe new-pixel-recipe.slp
```

Run from this example folder, or supply absolute paths. A guided training popup
and execution plugin are the next milestone; they are not part of this delivery.
Physical calibration/gauge validation and motion-based scan accumulation remain
future work. The original tool code and generated controls use [MIT](LICENSE).
The wheel retains its own code/license and requires NumPy, OpenCV and Pillow.
