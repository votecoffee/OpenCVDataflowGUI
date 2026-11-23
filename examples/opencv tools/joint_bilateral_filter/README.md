# Joint Bilateral Filtering

Open [joint_bilateral_filter.json](joint_bilateral_filter.json) and click **Run once**. The project smooths a noisy 16-bit image with a faint stripe and an 8,000-level step. The registered 8-bit color guide marks the same boundaries. Select each filter to inspect the result.

| Output | RMS error against the clean target | Measured step contrast | Measured stripe contrast |
| --- | ---: | ---: | ---: |
| Noisy input | 2,188.6 | 7,825.9 | 2,979.1 |
| Bilateral on source | 562.4 | 3,014.8 | 1,680.3 |
| Joint bilateral with guide | 191.8 | 8,061.1 | 3,097.9 |

The known step and stripe contrasts are 8,000 and 2,900 sample levels. The example's guide helps the filter retain those boundaries while smoothing the surrounding pixels. The selected branch writes the same filtered values inside the rectangle and preserves the original pixels elsewhere.

## Use the block

- Find **Joint bilateral filter** under **OpenCV → Extended Image Processing (ximgproc) → Filters**. Connect a monochrome or color **Guide** on the source's grid, or leave it disconnected for self-guided bilateral smoothing. Choose **Pixel grid** for images already registered by pixel index that were loaded separately. **Same frame** also checks known coordinate metadata.
- Set **Diameter**, **Sigma color**, **Sigma space**, and the **Border** dropdown. A diameter of zero derives the neighborhood from the spatial sigma. Color sigma uses normalized units for integer guides and stored sample units for float guides. Space sigma is in pixels.
- The default output keeps the source depth, encoding, ticket and grid. Select uint8, uint16 or float32 to convert the output. The filter preserves alpha, and an optional mask or region controls where filtered pixels are written. The full image supplies neighborhood context.

[OpenCV joint bilateral reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_filters.html)

The [clean target](assets/clean.png), source and guide are original synthetic assets covered by this repository's MIT license. Rebuild them and the saved project with `python build_example.py --output <folder>` in the application environment.
