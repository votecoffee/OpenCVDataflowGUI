# Multi-baseline stereo depth

Open [multi_baseline_depth.json](multi_baseline_depth.json) and run once.
Three calibrated pairs share a left camera and use 40, 80, and 120 mm
baselines. The supplied disparity rasters encode exact correspondence
for a synthetic raised foreground plane and background, including the
parts of the left view each baseline cannot see. They also include
one deliberately incorrect near-depth patch and one equal-support
disagreement. The reference depth connects only to the evaluation blocks.

| Result | Nearest surface | Consensus |
| --- | ---: | ---: |
| Reference pixels | 16,896 | 16,896 |
| Retained pixels | 15,872 | 15,944 |
| Coverage | 93.94% | 94.37% |
| Errors greater than 1 mm | 0 | 0 |

Both branches require two agreeing baselines. Consensus recovers 72
background pixels by discarding the false nearer measurement from the
120 mm pair. The 36 pixels with a tied vote remain unresolved. The
foreground's occlusion shadow has low support. The preview shows supported
pixels in green, retained disagreements in yellow, unsupported conflicts
in red, and ties in magenta. The rejection map uses bits 1, 2 and 4 for
baselines A, B and C.

For independently rectified pairs, use **Disparity to 3D cloud** to
recover each rectified camera-Z image. **Rectified stereo camera
geometry** extracts the rectified left intrinsics and the inverse
rectification rotation. Use these in **Register RGB-D with confidence**
to put every depth map onto the same original left-camera image grid.
Then use the **Consensus** surface policy in **Fuse registered depth
views** and inspect **Support Count**, **Rejected Views**, and
**Ambiguous Mask**. A rejection identifies disagreement with the
winning group. Visibility, matcher diagnostics, or independent depth
provide evidence for its cause. [OpenCV stereo rectification reference](https://docs.opencv.org/5.0/main_modules/modules_2stereo_2include_2opencv2_2stereo_8hpp.html)

The synthetic images and disparity controls are by OpenCVDataflowGUI
contributors (MIT). Regenerate them with
`python build_example.py --output <new-directory>`.
