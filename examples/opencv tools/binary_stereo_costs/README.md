# Inspect binary stereo match costs

Open the [example project](binary_stereo_costs.json) and run once. The two
diagnostic blocks compare center and symmetric Census on the same synthetic
rectified pair. The output image marks the three probes and the result
graphics include a cost curve. Change **Plot probe** to 1, 2 or 3 and rerun
to see that probe's curve. The **curves** output contains all candidates.

| Probe | What to look for |
| --- | --- |
| 1. Textured | The true 8 px disparity is the only zero-cost candidate. The next cost is 272 for center Census and 138 for symmetric Census. |
| 2. Flat | All 16 candidate costs are zero. The probe emits the tied disparities and withholds a match. |
| 3. Occluded | The right mask covers the candidate at 8 px. Four other candidates have costs, with gaps marking the unavailable footprints. The probe withholds a match. |

Connect a rectified pair and define whole-pixel probe coordinates on the left
image. The optional point input overrides saved probes. Use the same search
interval and window sizes as the dense binary matcher when comparing center
Census costs. Alternative descriptors compare with the local mean, compare
opposite pairs, or sample the axes and diagonals. Their bit counts differ,
so read the reported maximum cost when comparing variants.

The synthetic captures are original work by OpenCVDataflowGUI contributors
under the project's MIT license. Run the example builder with an empty
output directory to regenerate them. The dense stereo tools and descriptor
families are described in the [OpenCV stereo reference](https://docs.opencv.org/4.12.0/d2/d7f/namespacecv_1_1stereo.html).
