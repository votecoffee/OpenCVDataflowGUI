# Stereo exposure and color correction

Open [stereo_color.json](stereo_color.json) and run once. The synthetic
horizontal stereo rig uses a focal length of 160 px, a 50 mm baseline, and
a plane 1 m away. The known disparity is 8 px. The right image has reduced
contrast and a separate brightness offset in each color channel.

The graph rectifies both views and uses **Binary Census stereo disparity**
to seed correspondences. **Match stereo exposure and color** fits a
right-to-left gain and bias per RGB channel from the top rectangle. It
uses no pixels from the lower evaluation rectangle. The two **StereoBM**
branches use the same parameters and geometric support.

| Held-out result | Raw right image | Corrected right image |
| --- | ---: | ---: |
| Reference pixels | 9,540 | 9,540 |
| Valid disparity coverage | 95.24% | 98.11% |
| Mean absolute disparity error | 0.0851 px | 0.0356 px |
| Errors greater than 1 px | 0 | 0 |

The fit accepted 9,752 correspondences. Its mean absolute color residual
over usable pairs fell from 0.09285 to 0.00719 in normalized channel units
before output quantization. The numerical evidence is specific to this
controlled pair. The graph does not model or recover clipped detail,
different camera response curves, or spatially varying illumination.

Open the correction report to inspect gains, biases, and rejected pairs.
The fit mask and optional residual graphic use left-view coordinates.
The corrected image stays on the right grid. An optional right mask
limits both the correspondence footprints and the pixels changed.

The original synthetic rasters are by the OpenCVDataflowGUI contributors
(MIT). Rebuild them and the saved graph with
`python build_example.py --output <new-directory>`.
