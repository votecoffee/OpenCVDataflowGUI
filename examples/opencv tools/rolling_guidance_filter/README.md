# Rolling Guidance Filtering

Open [rolling_guidance_filter.json](rolling_guidance_filter.json) and click **Run once**. The synthetic plate has a coarse outline, a notch, a circular cutout, and 2 × 2 pixel surface texture. The project compares one pass and four passes, plus a branch that writes only inside a rectangle.

| Output | Flat-area texture standard deviation | Plate edge contrast | Cutout contrast |
| --- | ---: | ---: | ---: |
| Textured input | 18.26 | 118.45 | 93.38 |
| One pass | 11.80 | 118.31 | 92.91 |
| Four passes | 0.27 | 118.04 | 93.08 |

The known contrasts are 118 levels at the outer edge and 93 at the cutout. The four-pass result suppresses the fine pattern while recovering both contrasts on this synthetic target. It can also smooth small inspection features of a similar scale, so choose the neighborhood and iteration count with the feature sizes in mind.

## Use the block

- Find **Rolling guidance filter** under **OpenCV → Extended Image Processing (ximgproc) → Filters**. Each result guides the next pass over the original image. Set **Iterations** to zero to pass the image through with the selected output depth.
- **Diameter**, **Sigma space** and **Sigma color** set the spatial and intensity neighborhoods. Zero diameter derives the neighborhood from spatial sigma. Spatial sigma is measured in pixels. Integer inputs are processed on the [0, 1] scale, and float inputs use their stored intensity scale. **Border** selects extrapolation at the image edge.
- The default output preserves depth, encoding and alpha. Choose uint8, uint16 or float32 to convert the output. An optional mask or region restricts where filtered values are written. Each pass uses the full image as context.

[OpenCV rolling guidance filter reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_filters.html)

The input image and [clean target](assets/clean.png) are original synthetic assets covered by this repository's MIT license. Use `python build_example.py --output <folder>` in the application environment to regenerate the assets and project.
