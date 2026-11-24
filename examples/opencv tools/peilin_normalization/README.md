# Pei–Lin shape normalization

Open [peilin_normalization.json](peilin_normalization.json) and choose
**Run Once**. Compare **Compare originals** and **Compare normalized
shapes**. Two silhouettes come from the same polygon. The second was drawn
after an affine change that introduces translation, shear and unequal scaling.

| Comparison | Foreground intersection over union |
| --- | ---: |
| Original silhouettes | 38.9% |
| Normalized on a shared 200 × 200 canvas | 97.7% |

Corresponding vertices from the [known affine transform](truth.json) map
within two pixels of one another on the normalized grid. Nearest-neighbor
sampling keeps the example masks binary.

The block estimates geometry from intensity, a bright or dark threshold, or
a wired mask or region used as the silhouette. It warps the original image
with its depth and channel order preserved. The **Transform** output includes
both directions and can feed **Warp affine**. The **Geometry** output keeps
the mapping back to the source pixel grid.

The report records the estimator and whether the foreground has a usable
third-moment orientation. Symmetric shapes can center and scale with
ambiguous orientation. A stable moment calculation also handles
axis-aligned cases when the native eigenvector calculation is undefined.
Empty or collinear support produces a validation error. [OpenCV Pei–Lin reference](https://docs.opencv.org/5.0/extra_modules/ximgproc.html)

The two PNGs are original MIT-licensed assets. Run
`python build_example.py --output <directory>` to recreate them and the project.
