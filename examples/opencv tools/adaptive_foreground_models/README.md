# Adaptive foreground models

Open [adaptive_foreground_models.json](adaptive_foreground_models.json) and choose
**Run All**. Three OpenCV blocks receive the same lossless grayscale conveyor
replay. Inspect each block's **Ready**, **Status**, and **Mask** outputs. The
filmstrip shows the GMG mask once the model is ready.

| Block | First ready frame (zero-based) | Foreground on each part frame |
| --- | ---: | ---: |
| MOG | 1 | 480 of 480 pixels |
| GMG | 13 | 460 of 480 pixels |
| CNT | 10 | 480 of 480 pixels |

The two part positions occupy frames 22–23 and 24–25. GMG's 7-pixel median
kernel removes 20 corner pixels. Each model clears the mask on frame 26.
CNT also exposes a grayscale background estimate. These measurements are
specific to the supplied clip and settings.

Give each model clean frames when starting a new source. A mask or region limits
the reported foreground while learning uses the full frame. GMG's memory use
grows with image size and the number of retained colors per pixel. For an 8-bit
640 × 480 image with 64 colors per pixel, the native model estimate is 151 MiB.

The clip reuses the repository's original MIT-licensed
[empty-conveyor reference fixture](../background_reference/README.md).
Recreate it and the project with `build_example.py --output <directory>` in
the project's Python environment.
