# First CPU FoundationPose replay

The [completed-result examples](completed_runtime/README.md) add a separate
read-only Results workflow in dev4: exact executed RGB-D/mask/model, complete
original ranking cohort, stored first-surface fit samples, native colored
fly-around/sample selection and frozen exports. Two assisted real examples and
one full 252-candidate replay preserve prior results and all real rejections.
Six portable projects include immediate file-only cloud/image review.

This experimental pilot runs the actual pinned NVIDIA ONNX refiner/scorer on
one selected T-LESS instance with a **supplied visible ground-truth mask**.
It is not unassisted bin detection, a new network training run, GPU/CUDA parity,
official BOP evaluation or physical grasp qualification.

`trained.sfp` retains object 6's canonical mesh, original model origin, frame-3
RGB/depth/mask/calibration and fixed settings. Held-out frame 13 supplies its
own camera and measured inputs in `held_out_frame13/capture.npz`. Pose labels
enter evaluation only. The network has 252 unseeded rotation hypotheses, two
refinement passes, refiner batches of eight and one full-cohort scoring call.
All raw proposals remain in `report.json`. Weights are external, pinned and
verified; they are not part of this example collection.

| Evidence | Result |
| --- | --- |
| Top raw pose, finite-symmetry evaluation | 1.1994 mm translation / 2.6807° rotation error. |
| Top raw pose's measured support | 67.52%; contradiction 9.27%, support Z RMSE 2.30 mm. |
| Actual acceptance | Rejected: below unchanged 70% support gate. No accepted best pose. |
| Independent control | Analytic first-return/perspective and archive/origin tests; actual-model cohort/refiner batching checks. |
| Initial CPU elapsed time | About 502 s for this frame; development run, not controlled hardware performance qualification. |
| Different-object negative | Scene 3/frame 10, object-8 mask presented to object-6 model: no accepted pose, about 647 s. |

The score model is cohort-dependent. Scoring separate microbatches changes
logits; keep the complete hypothesis set together. Raw logits are not
probabilities. The fit check uses original measured depth and preserves unknowns,
even when the model's preprocessing filters/fills its own input. A correct raw
pose can therefore coexist with a rejected application result.

`overlay.png` shows the raw top proposal when there is no accepted pose;
its yellow text says **rejected**. Green vertices agree with measured depth,
red contradict free space, purple are occluded and orange are unknown/outside.
The model's hidden vertices in that image are explanatory graphics, not the
first-return samples used for acceptance. `fit.ply` is a standard RGB cloud
for native flyaround review; set units **m** and frame **camera**. It includes
the measured scene, derived model fit colors and original part-origin axes.

Reproduce in the isolated worker environment from the tool directory:

```powershell
python fetch_models.py .local/models
python replay_tless.py source .local/models .local/replay_new
```

The source corpus is pinned by selected-member hashes; the complete upstream
ZIP was not downloaded. The script does not supply annotated object poses to
registration. Canonical geometry/settings are shared, without choosing new
tolerances for this frame. The independent GPU renderer comparison and broader
multi-instance accuracy still need work. The second case deliberately supplies
an unrelated object's mask to check rejection of that proposal; it is not a
test of an independent instance detector.

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
WACV 2017. [Publisher/license](https://cmp.felk.cvut.cz/t-less/) and
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Mirror revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`; raw member hashes
are in `source/provenance.json`. Source and derived reference/results retain
CC BY 4.0. Code and model notices remain separate in [THIRD_PARTY.md](THIRD_PARTY.md).
No endorsement is implied.

## Open the saved result projects

Open `held_out_frame13/review_project.json` or
`different_object_negative/review_project.json`. Select the cloud for the
native RGB flyaround thumbnail/popout, and compare the source RGB and overlay.
These fast offline graphs load saved actual inference artifacts. They do not
require a GPU or the experimental worker package. Source blocks passing means
files loaded, not that the recorded pose passed acceptance; both reports are
rejected. Raw JSON evidence remains authoritative.

To rerun inference, use a separate environment with the experimental
`opencv-simple-foundation-pose[worker]` package, then the explicit commands
above. Host-side live inference/training integration is not implemented yet.
Do not install a second OpenCV wheel into the app environment for these reviews.
