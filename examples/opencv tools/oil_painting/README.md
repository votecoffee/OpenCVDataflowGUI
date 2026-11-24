# Oil painting texture filter

Open [oil_painting.json](oil_painting.json) and choose **Run Once**. Compare
the left edge of the plate in **Gaussian 9 × 9** and **Oil painting 9 × 9**.
The third branch writes the same histogram-filtered samples inside a rectangle
and keeps the original samples elsewhere.

| Image | Interior texture standard deviation | Largest one-pixel edge rise |
| --- | ---: | ---: |
| Source | 9.81 | 87.51 |
| Gaussian, sigma 2 | 0 | 18 |
| Oil painting, bin divisor 12 | 0 | 86 |

Measurements use 8-bit grayscale values. The histogram filter moved the
strongest edge one pixel toward the background on this fixture. Compare
[the clean reference](assets/clean.png) when tuning the radius and bin divisor
for inspection.

The block accepts 8-bit gray, BGR or RGB. It preserves straight alpha and
source channel order. A mask, region or their intersection controls which
pixels may change. Neighboring pixels outside the selection still contribute.

The default groups pixels by gray intensity, using OpenCV's native
`xphoto.oilPainting`. The installed OpenCV 5.0.0 build produced inconsistent
results when asked to convert to Lab or HSV internally. Those two choices
compute the first plane separately and use a deterministic local histogram
and color mean. Hue uses linear bins. [OpenCV xphoto reference](https://docs.opencv.org/5.0/extra_modules/xphoto.html)

The two PNGs are original MIT-licensed assets. Run
`python build_example.py --output <directory>` to recreate them and the project.
