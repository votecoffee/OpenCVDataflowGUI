# Bilateral texture smoothing

Open [smooth_textured_boundary.json](smooth_textured_boundary.json) and select
**Run Once**. The source is a synthetic textured plate with a known slanted
step. The filter runs once at radius 3 within the write rectangle. Separate
measurement regions sample the surface texture and fit the boundary before
and after filtering.

| Measurement | Source | Filtered |
| --- | ---: | ---: |
| Green-channel surface standard deviation | 8.54 | 0.80 |
| 10–90% fitted edge width | 1.11 px | 0.31 px |
| Fitted edge position at row 72 | 95.87 px | 96.01 px |

The true edge position at row 72 is 96 px. On this fixture, the filter reduces
the texture standard deviation by about 91% and locates the boundary within
0.1 px of its known position. The input PNG is an original synthetic fixture
covered by the repository license.

## Using the block

**Bilateral texture smoothing** exposes the native radius, iteration count,
transition sharpness (**Sigma alpha**) and range sigma (**Sigma avg**).
Negative sigma settings use the native automatic values. Range sigma uses
unit intensity. The block accepts gray or RGB/BGR images with optional
straight alpha. It preserves the input depth and color encoding. uint16 and
float64 use float32 working values, and floating intensities use `[0,1]`.

A mask or region limits writes to the chosen pixels. The working patch includes
enough surrounding source pixels for all configured iterations. Pixels outside
the selection and straight alpha samples retain their original values. Larger
windows and extra iterations increase the required work, so the block bounds
the working area and neighborhood work. [OpenCV bilateral texture filter source](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/bilateral_texture_filter.cpp)
