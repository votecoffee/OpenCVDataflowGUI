# Gradient-Based Image Quality

Open [gradient_quality.json](gradient_quality.json) and click **Run once**. The graph compares an original synthetic color chart against itself, a Gaussian-blurred copy, and a copy with seeded Gaussian noise. Select a comparison block to see its channel scores and the **Gradient similarity loss** preview.

| Compared image | GMSD, highest channel |
| --- | ---: |
| Identical | 0.000 |
| Gaussian blur | 0.221 |
| Gaussian noise | 0.340 |

GMSD measures the spatial variation of gradient similarity. Lower values indicate more uniform similarity, and an identical pair scores zero. Uniform changes can also produce low deviation, so use the map or a complementary metric when judging absolute agreement. In the map, values close to 1 indicate similar local gradients. The preview shows loss with a fixed color scale so results from different runs can be compared.

## Use the block with your images

- Feed two aligned 8-bit images into **Reference** and **Image**. With **Pixel grid** alignment, the block compares corresponding indices across independent captures. Select **Same frame** to also check known spatial metadata. Align the images upstream when needed.
- Choose grayscale or separate channel scores. Per-channel mode reports blue, green, red and alpha when present. Choose the maximum or mean channel deviation for the scalar output. Enable **Check threshold** to flag a failure when it exceeds **Maximum deviation**.
- Connect a mask or region to score only map cells whose full 2 × 2 source tile is selected. The gradients still use neighboring pixels. Crop the inputs first when the crop boundary must define the computation.
- The **Map** connector delivers the full native float32 similarity map. It retains its half-resolution grid, including OpenCV's rounding on odd dimensions. A map pixel at `(x, y)` is centered at `(2x + 0.5, 2y + 0.5)` in the inspected image. The map carries that image's ticket and transformed coordinate metadata.

[OpenCV QualityGMSD implementation](https://github.com/opencv/opencv_contrib/blob/5.x/modules/quality/src/qualitygmsd.cpp)

The reference chart is original to this repository and available under its MIT license. To rebuild the assets and graph, run `python build_example.py --output <folder>` with the application environment.
