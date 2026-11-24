# Tree-based Morse region repeatability

Open [the project](tbmr_repeatability.json) and choose **Run Once**. The
graph detects regions on the same nested bright-and-dark target at its
original size and at 1.5×. **Evaluate keypoint repeatability** uses the
known resize transform to pair centers one-to-one within 2 target pixels.
The review blocks draw those pairs.

| Detector | Query / resized detections | Matched centers | Repeatability | Median size ratio | 95th-percentile center error |
| --- | ---: | ---: | ---: | ---: | ---: |
| TBMR | 24 / 24 | 24 | 100.0% | 1.516 | 0.399 px |
| MSER | 26 / 13 | 8 | 61.5% | 1.508 | 0.243 px |

Repeatability divides matches by the smaller eligible population. The expected
size ratio is 1.5. TBMR size is the major-axis diameter and MSER size is the
diameter of a circle with the detected area. Each ratio compares a detector
against itself. The measurements apply to this synthetic target, the saved
parameters and the OpenCV 5.0 build. The two images use minimum areas of 32
and 72 pixels to reflect the 2.25× area change. MSER uses delta 5 and maximum
variation 0.35.

TBMR runs the native detector once per requested scale. It maps the center and
major-axis vector back to the input pixel grid and reports the original
detection level. The block retains detections from all processed levels.
Each detector raster can gain up to three replicated pixels on the bottom
and right to align with OpenCV's internal four-pixel grid. Mask and region
selections use the mapped centers.

The Python binding supplies the major-axis measurement and center. The block
draws that axis and leaves descriptor orientation unspecified. Input is
limited to 1,048,576 pixels and the pyramid to 4,194,304 padded pixels.
The maximum area and raster size are checked together against OpenCV's
32-bit moment range. Coarser levels stop when a dimension is smaller than
8 pixels or the minimum area exceeds the level's maximum.

The target is an original MIT-licensed synthetic asset. Rebuild with
`python build_example.py --output <directory>`.
[OpenCV TBMR reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1TBMR.html)
