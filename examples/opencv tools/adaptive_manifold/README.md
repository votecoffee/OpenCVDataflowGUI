# Adaptive manifold filtering

Open [adaptive_manifold.json](adaptive_manifold.json) and choose **Run Once**.
Select **Joint bilateral baseline** or **Adaptive manifold** to compare
filtered versions of the same noisy 16-bit signal. Both branches use the
registered color guide. The **Adaptive manifold in region** branch keeps
the input pixels outside the inspection rectangle.

The clean target has a vertical step of 8,200 intensity levels. The measured
step uses the two columns at the boundary, averaged across rows 14–79.

| Image | RMSE against the clean target | Measured step |
| --- | ---: | ---: |
| Noisy input | 2,148.1 | 8,201.0 |
| Joint bilateral | 146.4 | 8,225.8 |
| Adaptive manifold | 113.9 | 8,261.6 |

The block exposes the three controls in the OpenCV 5 Python factory. Spatial
sigma is measured in pixels. Color sigma is measured in normalized guide
intensity units. Outlier adjustment blends toward the source where the guide
is far from a fitted manifold. Tree height, PCA iteration count and RNG
selection are C++ settings in this release. [OpenCV filters reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_filters.html)

An unwired guide uses the source. A uniform guide uses OpenCV's spatial
recursive filter to avoid degenerate manifold splits. For other guides, the
block rejects reduced grids with fewer than two samples along an axis and
reports nonfinite native values as input errors. The mask and region select
output pixels after filtering. The block preserves alpha and defaults to the
source depth.

The generated PNGs, graph and builder are original MIT assets. Regenerate
them from the repository root with the example builder.

    python "examples/opencv tools/adaptive_manifold/build_example.py" --output "examples/opencv tools/adaptive_manifold"
