# BM3D denoising

Open [bm3d_denoising.json](bm3d_denoising.json) and choose **Run Once**. The
grayscale fixture has a 92-level step, small raised and recessed details, and
seeded Gaussian noise with sigma 13. Compare the **Basic** estimate and final
**Image** with the included [clean reference](assets/clean.png).

| Stage | RMSE against the reference, in intensity levels |
| --- | ---: |
| Noisy input | 13.08 |
| Portable hard threshold | 2.67 |
| Portable Wiener refinement | 2.46 |

The final portable image retains a 91.2-level step on the measured strip.
The block preserves image depth and writes only within a connected mask or region
when one is supplied. A Wiener-only graph accepts a wired basic estimate.

The example uses the portable DCT implementation of
[collaborative patch filtering](https://researchportal.tuni.fi/en/publications/image-denoising-by-sparse-3-d-transform-domain-collaborative-filt/).
The optional OpenCV backend uses its native Haar implementation on grayscale
uint8 or grayscale uint16 with L1 matching. The stock OpenCV wheel requires
the portable backend. [OpenCV BM3D build and API reference](https://docs.opencv.org/5.0/extra_modules/xphoto.html)

The isolated, enabled OpenCV 5.0.0 build measured final RMSE of 3.34 levels
for uint8 L2 and 2.58 for uint16 L1, expressed on the 8-bit scale. Separate
stage calls matched the combined call for both native depth and norm pairs.

The two PNGs are original MIT-licensed assets. Run
`build_example.py --output <directory>` to recreate them and the project.
