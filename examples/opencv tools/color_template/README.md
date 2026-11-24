# Joint color template matching

Open [compare_color_and_gray.json](compare_color_and_gray.json) and select
**Run Once**. The two keyed inserts have identical grayscale pixels. Their
grayscale template scores both round to **1.0000**. Joint color matching
selects the warm insert at **[149, 67]** with a native peak of **0.3946**.
The cooler insert at **[28, 32]** scores **0.2447** in that search image.

Open [reject_wrong_color.json](reject_wrong_color.json) for the negative case.
With only the cooler insert present, the strongest color peak is **0.3017**,
below the configured minimum of **0.35**. The grayscale branch still passes.
The color block reports the rejected location and withholds its pose. The
PNG images are original synthetic fixtures covered by the repository license.

## Using the block

Wire the color search image and a tight color crop at the same scale. **Match
color template** uses [OpenCV quaternion phase correlation](https://docs.opencv.org/4.13.0/da/d17/group__ximgproc__filters.html)
across the three working color channels. Tagged RGB, HSV and Lab are interpreted
as BGR for the calculation. Alpha is ignored in the calculation and preserved
in the source image output.

Wire a search mask or region to restrict where candidate centers may fall.
The float64 response preserves the scores for all complete placements. Each
response pixel indexes a template top-left in source coordinates. The result
view draws the strongest eligible bounds and the accepted pose can drive
pose-relative inspection. Optional subpixel refinement uses the eligible
neighborhood around an interior peak.

Set the minimum score from representative good and wrong-color samples.
The peak depends on the template and the entire search image. For larger
frames, use **Crop** before matching. The native transform is limited to
1,048,576 padded pixels, and spectra with undefined correlation report a
diagnostic.
