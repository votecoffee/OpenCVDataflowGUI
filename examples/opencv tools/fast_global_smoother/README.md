# Fast Global Edge-Preserving Smoothing

Open [fast_global_smoother.json](fast_global_smoother.json) and click **Run once**. The synthetic inspection image contains a 130-level vertical step, a smaller raised feature, and seeded noise. The project compares self-guided smoothing, a registered color guide, and a region that limits where the result is written.

| Output | RMS error against the clean image | Standard deviation in a flat patch |
| --- | ---: | ---: |
| Noisy input | 10.953 | 11.073 |
| Self-guided smoothing | 5.263 | 0.783 |
| Registered color guide | 0.000 | 0.000 |

In this fixture, the registered guide recovers the original 130-level step and smaller feature exactly in 8-bit output. The selected branch uses the same filtered values inside the rectangle and the original noisy pixels outside it.

## Use the block

Under **OpenCV → Extended Image Processing (ximgproc) → Filters**, choose **Fast global smoother**. With **Guide** disconnected, the source supplies the edges. To guide from a separately loaded image that is already registered pixel-for-pixel, set **Guide alignment** to **Pixel grid**. The default **Same frame** checks known coordinate metadata as well as dimensions.

- **Lambda weight** controls smoothing strength. **Sigma color** controls how readily smoothing crosses edges in the 8-bit guide. **Lambda attenuation** and **Iterations** control the successive passes.
- The output keeps the input encoding, grid, ticket and depth. The source supports uint8, int16, uint16, and float32. The block preserves alpha without smoothing it. For edge weights, signed and unsigned integer guides map their full ranges to 8-bit, and float guides use [0, 1].
- A mask or region selects where to write the result. The smoother uses the full image and guide as context, including around selection boundaries.

[OpenCV fast global smoother reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_filters.html)

The chart, guide and [clean target](assets/ground_truth.png) are original synthetic assets covered by this repository's MIT license. Rebuild the project and images with `python build_example.py --output <folder>` in the application environment.
