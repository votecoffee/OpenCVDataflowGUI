# DCT denoising

Open [dct_denoising.json](dct_denoising.json) and click **Run**. Compare **Simulated
sensor image**, **Whole-frame denoising**, and **Denoise the inspection area** in
their Results tabs. The last branch writes only inside the circle. Zoom in on
the diagonal line inside the rectangle to see the tradeoff between noise and
small features.

**Noise sigma** is the expected noise standard deviation on a 0–255 scale.
Use the same setting for equal relative noise in 8-bit, 16-bit and normalized
floating-point images. Zero bypasses filtering. **Patch size** is an even
length from 2 to 32 pixels that fits both image dimensions. Larger patches
need more work per pixel. The block keeps color encoding, alpha, image depth,
and spatial and ticket metadata. A mask and a region restrict writes to their
intersection. The source around each selected pixel still contributes to its
estimate.

The block calls OpenCV's sliding-patch DCT filter on overlapping tiles to
limit native patch storage. Each tile includes all source windows contributing
to its result. The wrapper gives OpenCV a disposable trailing row and column
so it computes estimates for the actual image edges.

| Measured on this fixture | Noisy | Denoised | Clean reference |
| --- | ---: | ---: | ---: |
| Whole-image RMS error | 12.04 | 4.03 | 0 |
| Blue-channel standard deviation in a flat patch | 12.06 | 0.80 | 0 |
| Mean step-edge contrast | 90.78 | 90.05 | 90.00 |

The strongest step stayed between columns 55 and 56, using zero-based indices.
The circle branch produced the same pixels as the whole-frame branch inside
the circle and preserved the original samples outside it.

The clean and noisy PNGs are original MIT-licensed fixtures. The generator
adds seeded Gaussian noise with a standard deviation of 12 levels. For the
native API, see [OpenCV xphoto documentation](https://docs.opencv.org/5.0/extra_modules/xphoto.html).
