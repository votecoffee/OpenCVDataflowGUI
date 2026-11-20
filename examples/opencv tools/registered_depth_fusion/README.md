# Registered depth fusion

Open [registered_depth_fusion.json](registered_depth_fusion.json) and
run once. The two depth rasters are already aligned. One uses
millimeters and one uses meters. Each has a confidence image. The
block fuses pixels within 60 mm of the nearest trusted surface.

On the background, the 2,010 mm and 1,970 mm measurements receive
weights of 1 and 1/3. Their weighted depth is 2,000 mm. Single-view
coverage also fills complementary holes. The shared hole remains
invalid. A false close return falls below the minimum confidence
and is excluded.

The foreground strip is two pixels wide and 64 pixels long. One
view sees the strip at 1,250 mm. The farther view sees the background
at that location. All 128 strip pixels retain 1,250 mm and appear
yellow in **Fusion support** because they are both valid and disputed.
Green marks fused pixels without an eligible farther surface.

| Result | Value |
| --- | ---: |
| Fused pixels | 13,804 of 13,824 |
| Coverage | 99.86% |
| Farther samples rejected on the strip | 128 |
| Pixels missing in both views | 20 |
| Mean absolute error against known depth | 0.336 mm |

The report, exact masks, confidence and support count are available
in Debug and Lean. Raising **Minimum Support** to two removes the
single-view strip and demonstrates the coverage gate.

The range images, confidence maps and reference are original synthetic
assets by the OpenCVDataflowGUI contributors (MIT). Rebuild in a fresh
directory with `python build_example.py --output <directory>`.
[OpenCV's metric depth registration API](https://docs.opencv.org/5.0/main_modules/ptcloud_rgbd.html)
