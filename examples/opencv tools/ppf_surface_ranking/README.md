# Rank 3D surface matches in clutter

Open [ppf_surface_ranking.json](ppf_surface_ranking.json) and run it. The
scene contains two partly occluded copies of an asymmetric curved panel
among 200 clutter points. The fixture records the generating poses in its
[builder](build_example.py).

| Branch | Expected result on OpenCV 5.0.0 |
| --- | --- |
| Review native vote order | Emits the five highest-vote candidates. The instance on the right ranks first. |
| Refine and rank on scene support | Keeps both true instances and ranks the more visible instance on the left first. Six of the eight evaluated candidates fail geometry checks. |

Select **Refined geometry** in the match block to enable verification.
Each refinement pass fits local reciprocal pairs inside the refinement
distance. The scoring radius defines model coverage independently of
unrelated points elsewhere in the scene. The score multiplies model
coverage by one minus the ratio of reciprocal inlier RMSE to the scoring
radius, clamped at zero. Both distances use the common cloud units. The
match keeps the highest-scoring fit that passes the coverage, inlier count,
and RMSE limits, including the original pose if its fit is better.

Accepted poses are checked in score order for nearby model centers and
similar orientations. The report lists rejected and suppressed candidates,
their seed and per-pass measurements, and which fit supplied each emitted
pose. The preview colors the accepted matches over the gray scene.
Geometric verification considers a bounded list of native hypotheses, so the
PPF sampling and candidate limit determine which poses reach this step.
[OpenCV PPF detector reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1ppf__match__3d_1_1PPF3DDetector.html)

The two PLY files are original OpenCVDataflowGUI contributions under the
repository's MIT license.
