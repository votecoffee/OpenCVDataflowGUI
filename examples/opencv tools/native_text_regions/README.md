# Native text components and groups

Open [native_text_regions.json](native_text_regions.json) and choose **Run Once**.
**Group horizontal text** returns three boxes around INLET, OUTLET and VALVE.
They cover dark-on-light, light-on-dark and lower-contrast printing. **Detect
components** shows 25 character candidates in the same source frame. Inspect
the grouped boxes in the filmstrip.

| Label | Intersection over union against the glyph bounds |
| --- | ---: |
| INLET | 0.93 |
| OUTLET | 0.93 |
| VALVE | 0.93 |

The [reference bounds](truth.json) come from the glyph pixels of the original
synthetic chart. The labels are provided as reference data. The block detects
regions and leaves recognition to an OCR block connected to **Regions**.
Horizontal grouping can combine words on the same line. The qualified OpenCV
build supports horizontal grouping.

Component mode offers grayscale, RGB/lightness/gradient and HSV/gradient
channels with configurable polarity. Both modes offer the two classifier
thresholds and area bounds. Masks and regions filter the returned boxes by
their centers. Too few detections or truncated results produce a failed
judgment with the available regions and a reason. [OpenCV scene text detection reference](https://docs.opencv.org/5.0/extra_modules/text_detect.html)

The app bundles the pinned OpenCV 5.0.0 NM1 and NM2 classifiers. The
[classifier provenance and license](../../../src/opencv_dataflow/models/text_er/SOURCES.md)
list both hashes. The report identifies the classifiers used, including local
overrides. The synthetic chart is MIT-licensed. Run
`python build_example.py --output <directory>` to recreate it and the project.
