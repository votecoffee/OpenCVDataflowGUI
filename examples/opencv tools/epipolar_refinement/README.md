# Epipolar correspondence refinement

Open [epipolar_refinement.json](epipolar_refinement.json) and run once. The
recipe supplies 32 ordered pixel pairs and the fundamental matrix of a known
stereo rig. The left and right preview images show the noisy observations.
The **Score and refine epipolar correspondences** block scores each pair,
corrects the plausible matches, and forwards them to **Check epipolar
correspondences** for a separate point-to-line check.

| Check | Result |
| --- | --- |
| Accepted pairs | 29 of 32 |
| Rejected input indices | 5, 19, 28 |
| Sampson RMS on the accepted pairs | 0.612 px before correction, under 10⁻⁹ px after |
| Position RMS against clean projected points | 0.837 px before correction, 0.717 px after |

The graph uses two different pinhole image grids and an oblique relative
pose. The [fixture generator](build_example.py) projects the clean points,
adds seeded subpixel noise, and injects the three bad matches. The images,
arrays and project are original synthetic fixtures by the OpenCVDataflowGUI
contributors under the repository [MIT license](../../../LICENSE).

Select the refinement block to view each side's correction preview. Red
marks rejected observations. Green shows accepted points and the move from
the original to the corrected position. Each drawable layer follows its
own image grid. **Indices** links emitted pairs to their original rows.
**Inliers** and the per-pair report stay in input order. **Evaluate** mode
scores the input and emits the accepted coordinates unchanged.

For other projects, supply a left-to-right fundamental matrix and point
coordinates expressed in the matching two pixel grids. Use the
square-root Sampson error in pixels to gate raw correspondences and the
per-image shift limit to bound how far correction may move each point.
The block preserves the recorded left and right frame payloads. A low
epipolar residual confirms the geometric constraint for a pair. Use
appearance, IDs or another independent cue when correspondence identity
needs verification. [OpenCV epipolar geometry reference](https://docs.opencv.org/doc/doxygen/html/d2/d48/group__d__projection.html)
