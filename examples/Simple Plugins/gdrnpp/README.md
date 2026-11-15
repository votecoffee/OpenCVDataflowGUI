# SimpleGDRNPP setup and real saved reviews

Open `frame3/review_project.json`, `frame13/review_project.json` or
`wrong_object/review_project.json` for fast offline review with the native RGB
flyaround thumbnail/popout. These load recorded actual GDRNPP results; source
blocks passing means successful file reading. **All recorded poses are rejected**
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

## Attribution

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**, WACV
2017. [Publisher and CC BY 4.0 license](https://cmp.felk.cvut.cz/t-less/).
Retained metric arrays, supplied rectangles, overlays and colored fit clouds
are modifications. No endorsement is implied. Pose ground truth is used only
for the saved evaluation metrics. Public GDRNPP source/model terms, authors and
dependency notices are recorded in `THIRD_PARTY.md`; weights stay external.
