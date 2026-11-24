# PCAFlow and SimpleFlow

Open [pca_simple_flow.json](pca_simple_flow.json) and choose **Run All**. The
second frame moves an original texture 3 pixels right and 2 pixels up. Both
blocks report `[dx, dy]` in pixels on the first frame's grid. Inspect the
direction-and-magnitude previews, vectors, and optional round-trip masks.

| Method | Mean endpoint error on the interior |
| --- | ---: |
| PCAFlow | 0.030 px |
| SimpleFlow | 0.001 px |

The fixture wraps at the frame edges. The comparison excludes a 16-pixel edge
strip and measures Euclidean distance to `[3, -2]` on the remaining grid.

- PCAFlow uses the fixed configuration exposed by the OpenCV 5 Python
  factory. Its native basis needs images at least 72 pixels wide and 56 high.
- SimpleFlow exposes pyramid depth, matching and search radii, occlusion
  threshold, and filtering controls. A large frame or neighborhood can exceed
  its processing budget. Crop before the block to reduce the solve area.

The two bundled PNGs are original MIT-licensed assets. Recreate the images and
project with `build_example.py --output <directory>` in the app environment.
