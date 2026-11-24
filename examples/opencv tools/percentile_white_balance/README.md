# Percentile white balance

Open [percentile_white_balance.json](percentile_white_balance.json) and choose
**Run Once**. Four gray steps have a color cast, and the first and last rows
contain deliberately dark and bright outliers. The filmstrip shows the exact
percentile branch with 2% trimmed from each tail.

| Method | Darkest step (B/G/R) | Brightest step (B/G/R) |
| --- | --- | --- |
| Exact, no trim | 36 / 66 / 48 | 147 / 159 / 156 |
| Exact, 2% trim | 16 / 16 / 16 | 235 / 235 / 235 |
| OpenCV histogram, 2% trim | 16 / 16 / 16 | 233 / 233 / 233 |

The trimmed branches make all four steps neutral and clip the outlier rows to
the configured 16–235 output range. The exact method reports the two fitted
intensity values for each channel. Input and output limits use the source
sample units. The depth defaults are 0–255, 0–65535 and 0–1 respectively.

**Exact percentiles** trims whole samples from the sorted ends of each channel.
A mask, region or their intersection can fit the stretch from a reference
area. The fitted stretch applies to the whole frame. A channel with coincident
fitted endpoints keeps its original samples within the output limits.

**OpenCV histogram** uses native `xphoto.createSimpleWB` on whole 8-bit and
float32 images. Its quantized histogram endpoints can land short of the
requested range, as shown by the brightest step above. The output is clipped
to the configured limits. [OpenCV SimpleWB reference](https://docs.opencv.org/5.0/extra_modules/xphoto.html)

The original chart is MIT-licensed. Run
`python build_example.py --output <directory>` to recreate it and the project.
