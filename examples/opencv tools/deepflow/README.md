# DeepFlow optical flow

Open [deepflow.json](deepflow.json) and choose **Run All**. The first frame
primes the sequence. The next frame moves a textured patch five pixels right
and two pixels up. The final frame moves it again and places an occluder
over part of the patch. Use the block's Results tab to compare **Visualization**,
**Consistency mask** and **Consistency error**.

**Flow** stores float32 `[dx, dy]` on the previous image grid.
The consistency mask compares that flow with a separate estimate in the reverse
direction. Black marks excessive disagreement or unevaluated pixels. The
floating-point error image uses NaN where a round trip is unavailable. A
mask or region limits reported results while the solver uses the full image.

On the final pair, mean endpoint error in the visible test area was **0.0473
pixels**. All 406 sampled visible points agreed within the saved 0.05-pixel
tolerance. The check flagged **580 of 1,110 hidden points**. The remaining
530 hidden points also agreed, so the mask measures motion consistency.
Ground-truth visibility in this example comes from the rendered rectangles.

The measured previous-frame slices are `y=36:65, x=41:55` for visible
points and `y=33:70, x=63:93` for the hidden interior. The occluder in the
final image starts at `[66,29]` and spans `34 × 41` pixels. Hidden truth
maps that area back by the known shift and excludes a two-pixel boundary.

This block uses the solver defaults in OpenCV 5's Python binding. The
app exposes the shared temporal, selection, consistency and preview controls.
Unsigned 16-bit and normalized floating-point images map to the native
0–255 intensity scale in float32. The block bounds the estimated native
work arrays independently of its sequence memory setting.

All three images are original MIT-licensed synthetic fixtures. Rebuild them
and the project with `build_example.py`.
[OpenCV DeepFlow reference](https://docs.opencv.org/5.0/extra_modules/optflow.html)
