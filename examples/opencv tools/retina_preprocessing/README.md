# Retina contrast and motion

Open [retina_preprocessing.json](retina_preprocessing.json) and choose
**Run All**. Each activation updates the same temporal model. The first five
frames warm it up. The block preview shows **Detail** (Parvo) and the filmstrip
shows **Motion** (Magno).

| Zero-based frame | What's in the clip | Mean raw motion |
| --- | --- | ---: |
| 11 | Stationary low exposure | 0.57 |
| 12 | Whole-frame exposure rises by 45 levels | 56.82 |
| 29 | Stationary after adaptation | 0.019 |
| 31 | Bright part moving across the right side | 8.49 |

Per-frame motion normalization is disabled so the raw values are comparable
over time. Parvo retains scene detail across the exposure step. Magno responds
to exposure changes and moving objects. It reports a temporal signal in native
units. A mask or region limits the published pixels while adaptation sees the
full frame. Reset or video rewind starts a fresh warmup.

The block keeps the original grid and exposes the
[OpenCV Retina filter controls](https://docs.opencv.org/5.0/extra_modules/classcv_1_1bioinspired_1_1Retina.html).
An optional color mosaic applies to color input. Black values are floored at
0.0001 on the native 0–255 input scale to keep adaptation finite.

The original MIT-licensed FFV1 clip and project can be recreated with
`build_example.py --output <directory>` in the app environment.
