# SimpleGDRNPP setup and real saved reviews

Open `frame3/review_project.json`, `frame13/review_project.json` or
`wrong_object/review_project.json` for fast offline review with the native RGB
flyaround thumbnail/popout. These load recorded actual GDRNPP results; source
blocks passing means successful file reading. **All baseline recorded poses are rejected**
by unchanged measured-depth gates. The wrong-object case deliberately presents
an object-8 rectangle to the object-6 model. No automatic detector or physical
grasp acceptance is claimed.

For the native setup popup, explicitly load `plugin.py`, then open
`setup_project.json` and click **Set up trained object**. The portable recipe
retains the complete reference RGB-D/CAD/calibration/rectangles/symmetries and
dataset credit. It contains neither weights nor pose labels. Open/reference
inspection needs the separately installed worker but does not load its neural
model. New-object network training is not implemented; this checkpoint is for
T-LESS object 6 only.

## Isolated runtime

Install the separately supplied `opencv-simple-gdrnpp[worker]` package and its
original geometry dependency in a separate Python 3.11+ environment. Choose
that Python using **Choose runtime** in the popup, or set
`SIMPLE_GDRNPP_PYTHON` before launching the app. The trusted runtime profile is
an app/user setting, not executable code selected by a saved project. Keep
PyTorch/timm and alternate OpenCV wheels outside the host environment.

Run `python fetch_model.py NEW_MODEL_FOLDER` to explicitly download and verify
the selected author-published checkpoint (about 410 MB). Choose this external
folder in Setup, then preview the retained reference or a separate calibrated
capture. Edit boxes by drawing or changing coordinates; scroll to zoom and drag
to pan. Fit gates display millimeters and percentages. **Stage setup** proposes
a new sealed asset; inspector **Apply configuration** commits its path. Cancel
discards popup edits and cancels pending previews. Original recipes stay intact.

For live graphs choose **Connected RGB-D** mode and connect all five inputs:
uint8 color image, registered physical depth, camera, detection records and an
explicit capture ID. Camera data declares `matrix`, `image_size`, `frame` and
optional zero distortion. Detection data is a list of `identifier` / `box`
XYXY rectangles with exclusive upper bounds. Depth must already share the color
pixel grid; the tool does not synchronize or register sensors. The popup can
capture connected inputs as a reference or preview them privately.

Only depth-accepted poses reach `pose` / `poses`; raw proposals remain in
`report`. The cloud is a derived review containing measured scene, predicted
camera-visible CAD and pose axes. Green supports measured depth, red contradicts
it, purple is behind it and orange is unknown/outside the selected rectangle.
Select the `cloud` output for native flyaround. Each block/draft owns a bounded
isolated worker and keeps its model/recipe loaded between calls. Turning
graphics off skips the overlay/cloud render work. Cancellation retires that
worker so an older computation cannot publish into the current draft.

## Optional local depth alignment

`setup_icp_project.json` opens a separate opt-in setup with bounded robust
projective ICP. In the popup enable depth refinement and choose **Robust
projective ICP (local 6D)**. The original `setup_project.json` and baseline
assets/results are unchanged. Use the same separately installed worker.

`projective_refinement/frame3/review_project.json`,
`projective_refinement/frame13/review_project.json` and
`projective_refinement/wrong_object/review_project.json` retain full overlays,
native RGB flyaround clouds, original RGB and explicit decision labels. The
comparison image and summary JSON retain all three development cases.

All use the same 5 mm / >=70% support / <=10% contradiction gates. Frame 3
reaches 76.27% support / 10.33% contradiction and remains rejected. Frame 13
reaches 72.05% / 5.48% and passes these depth gates, but its translation error
against annotated pose **worsens from 1.067 to 4.251 mm** (rotation improves
from 3.247 to 2.386 degrees). The wrong-object control remains rejected.
Published annotated poses also fail the raw-depth gates on both positives;
sensor/model disagreement is not solely network pose error. Those labels enter
evaluation only. This uses the same depth for refinement and verification,
not an independent measurement or grasp qualification.

The original NumPy/OpenCV local solver uses first-return CAD, measured rays at
strides 4/2/1, robust point-to-plane updates, explicit rank and motion bounds.
It retains raw pose and rejects updates that worsen any gate violation. No
extra Open3D dependency is needed. See the [point-to-plane mathematical
background](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).
Sampled positive refinement took about 6.4-7.2 seconds on CPU; no full-cycle
speed or broad accuracy claim is made. Independent scenes, GPU profiling and
actual new-object training remain open.

## Broader real scene extension

`scene_extension/gallery.html` retains raw/local-alignment comparisons for all
29 metadata-selected cases from 18 new scenes. Twelve positives in scene 6
improve from 3 to 9 depth-accepted poses; all nine are within the development
5 mm / 5 degree annotation criterion. All seventeen target-absent controls
remain rejected, including a lookalike with 78.29% support but 12.15%
contradiction. Settings and checkpoint remain unchanged. No gate was loosened.

Four `scene_extension/reviews/*/review_project.json` projects compare raw and
aligned native colored flyaround clouds, full source RGB and pose/fit images.
They retain an improvement, remaining failure, worst accepted annotation error
and strongest absent control. File-source PASS is not pose acceptance. All
29 cases remain in summary/evaluation, with public T-LESS CC BY 4.0 attribution.

The positive views are correlated, from one additional arrangement and at least
91.4% visible. Heavy occlusion, new objects, detection and picking remain open.
Fitting/verification reuse depth and sometimes worsen annotation error. See
`scene_extension/README.md` for commands, timing scopes and remaining limits.

## Attribution

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**, WACV
2017. [Publisher and CC BY 4.0 license](https://cmp.felk.cvut.cz/t-less/).
Retained metric arrays, supplied rectangles, overlays and colored fit clouds
are modifications. No endorsement is implied. Pose ground truth is used only
for the saved evaluation metrics. Public GDRNPP source/model terms, authors and
dependency notices are recorded in `THIRD_PARTY.md`; weights stay external.

## Geometry execution performance

The updated separate worker uses bounded batched geometry for fitting and
predicted review surfaces. It needs no new training, recipe or host dependency.
Reset/restart an already running worker after updating its package.

`performance/README.md` and `performance/comparison.png` retain three paired
warm CPU measurements on four fixed cases: alignment plus final fit improves
about 5.0-8.5x, excluding neural inference/startup/transport/review graphics.
All 32 frozen poses, decisions and native review outputs are unchanged on the
tested installation. All earlier files above remain as historical evidence.
Broader accuracy, heavy occlusion and complete machine cycle qualification
remain open; these numbers are not a general inference-rate guarantee.

`worker_cycles/README.md` adds actual cold/warm subprocess measurements with
graphics on/off and a runnable `benchmark_worker.py`. Prepared rays and one
shared render per image/cloud review preserve all 32 frozen results. Paired
alignment improves another 1.26-1.78x; the separate sequential whole-call runs
do not establish a consistent overall gain. Updated warm examples took about
1.3-2.8 seconds, and cold examples 11.5-16.4 seconds, with fixed settings.
Profiles now separate worker computation, graphics, model setup and client
transfer/cleanup. Keep persistent model reuse; restart/reset an updated worker
and restart the app to load its updated light bridge. Recipes are unchanged.
