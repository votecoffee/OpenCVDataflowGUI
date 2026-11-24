# Fuzzy image smoothing and inpainting

Open [fuzzy_restoration.json](fuzzy_restoration.json) and choose **Run Once**.
Compare the smoothed upper ramp and the three lower repair branches against
[the clean reference](assets/clean.png). The filmstrip shows the iterative
repair of the 7 × 7 hole.

| Comparison against the clean ramp | Input RMSE | Output RMSE |
| --- | ---: | ---: |
| Smoothing with radius 3, whole image | 7.06 | 1.41 |
| One-step repair at radius 4, inside the hole | 99.92 | 1.24 |
| Multi-step repair from radius 2, inside the hole | 99.92 | 1.24 |
| Iterative repair from radius 2, inside the hole | 99.92 | 0.94 |

Smoothing uses a triangular or custom separable membership kernel. Custom
profiles use an odd number of weights, zero endpoints, a unit center, and
overlapping weights that sum to one. A mask or region limits output writes.
Neighbors outside the selection still contribute. Padding completes the last
fuzzy-grid interval and preserves constant intensities at the edges.

One-step inpainting uses the chosen radius. Multi-step raises the radius until
all components have support. Iterative repair also reuses supported estimates
from earlier passes. Both adaptive methods honor **Maximum radius**. An
incomplete repair reports a validation error. All three preserve measured
pixels outside the selection, and **Repair mask** marks synthesized pixels.
[OpenCV fuzzy image processing](https://docs.opencv.org/5.0/extra_modules/f_image.html)

The PNGs are original MIT-licensed assets. Run
`python build_example.py --output <directory>` to recreate the project.
