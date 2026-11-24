# Sparse-to-dense optical flow

Open [sparse_to_dense_flow.json](sparse_to_dense_flow.json) and choose
**Run All**. The replay contains two textured panels. The dark panel moves
4 pixels down and the bright panel moves 3 pixels up. The first frame supplies
the reference grid. The second produces a direction-and-magnitude preview,
the underlying pixel displacements, and a forward/backward agreement mask.

Compare the two previews around the center seam. On this fixture the default
settings preserve the boundary, with mean endpoint error of about 0.08 px
within 8 px of either side. The coarse grid without smoothing has about 6.70 px
mean error on the dark side of that strip, even though the interiors of both
panels are accurate. The outer 18 rows are excluded from these measurements
because the synthetic motion wraps at the top and bottom.

Grid step, neighborhood size, weight falloff, and post-smoothing shape the
interpolation. The requested neighborhood must fit the grid and the set of
valid tracks. The agreement mask measures a round trip in the estimated flow.
Use known reference motion when you need to measure accuracy.

The bundled images are original MIT-licensed assets. Recreate the project and
images with `build_example.py --output <directory>` in the app environment.
