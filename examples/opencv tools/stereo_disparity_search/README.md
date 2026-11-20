# Stereo disparity search and support

Open [stereo_disparity_search.json](stereo_disparity_search.json) and run once.
Both branches use the same rectified synthetic images, calibrated rig and
two masks. A rectangle bounds the left inspection area. A second mask marks
an unavailable patch in the right image. The known scene is a flat surface
at 3 m with 8 pixels of horizontal disparity.

| Result | Manual search | Depth-based search |
| --- | ---: | ---: |
| Native search interval | 0–63 px | 5–20 px |
| Native search count | 64 | 16 |
| Pixels with full matching support | 34,865 (60.5%) | 44,269 (76.9%) |
| Median disparity | 8 px | 8 px |

The requested depth interval is 2.4–4.4 m. The block maps those endpoints
through the calibrated Q matrix, accounts for disparity sign and principal
point offset, and rounds the native search count to a multiple of 16.
The narrower branch reprojects to a median depth of 3 m. The
[project generator](build_example.py) uses the app's original synthetic
stereo source and requires no image assets or hardware.

Connect **Rectification** to enable the calibrated valid rectangles and to
derive a range from depth. Enter near and far limits in the record's units.
Alternatively, use the manual pixel controls. For a pair with irregular
support, connect **Left mask** and **Right mask** in their corresponding
rectified grids. The block starts with OpenCV's valid ROI, checks the signed
right-hand boundary, and requires the full block footprint for every
candidate disparity, including around holes in either mask.

**Search report** shows the predicted and actual search ranges, the
rectangular ROI before mask holes, and the fraction with complete support.
**Support mask** shows where matching is geometrically supported. **Valid
mask** also applies the matcher's validity and confidence checks. Disparity
is NaN outside valid support, and previews and confidence are zeroed there.
BM receives the per-view ROI rectangles. SGBM computes on the full-size
grids and filters the outputs by support.
[OpenCV stereo API](https://docs.opencv.org/5.0/main_modules/modules_2stereo_2include_2opencv2_2stereo_8hpp.html)
