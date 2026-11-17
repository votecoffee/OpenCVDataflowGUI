# Review a completed 3D localization

Version 0.1.0.dev21 adds **Review completed 3D localization…** on the inspector's
Results page, separate from **Train 3D locator…**. It uses the host's native
CloudViewer with RGB point colors, yaw/pitch, point size and Flyaround controls.
The node's primary preview remains the native `fit` cloud.

1. Run in Debug with input/output previews retained. Open Results review to
   atomically capture the completed inputs/outputs and their model binding.
2. Start with all accepted fits. Gray points are the sampled measured scene;
   colored fitted points are expected model geometry, not new sensor returns.
3. Select an accepted instance or a retained rejected candidate. Read its actual
   support, RMSE, geometric/depth/projective/boundary gate evidence and limits.
   Toggle sampled scene, accepted pose axes, competing alternative and fit layers.
4. Choose **Exact consumed scene** or **Exact consumed depth_scene** to inspect
   the original typed cloud with its colors/normals/pixel mapping. These clouds
   differ from the sampled search/verification scene. Fit-layer switches are
   disabled while an exact input is displayed.
5. Orbit or enable Flyaround. Export frozen review writes a new directory with
   the selected display, native arrays, exact original recipe and full report.
   Close review without changing training or inspector drafts.
6. Later editor publications mark the view frozen. Capture latest explicitly
   replaces it. Active/error/unretained capture refuses to mix runs and keeps
   a prior frame clearly labeled. Before a completion, export is disabled.

Green expected samples have recorded support; red lack support; blue-gray are
self-hidden; magenta contradict known free space. RGB axes mark accepted
model-frame origin/orientation. Orange geometry is rejected or competing; it
never becomes a detection. Overlap rejection and accepted score ordering remain
unchanged. At most eight rejected candidates were retained by execution; the
viewer cannot invent the discarded candidates. Geometric/projective scores are
not calibrated probabilities and an accepted pose can still be physically wrong.

No matching, PPF/ICP, normal estimation, CAD sampling, depth rasterization or
boundary verification is run when capturing/reviewing. Fit-layer rendering only
transforms frozen model samples using recorded poses and support masks. Per-ray
depth/boundary maps are explicitly unavailable because runtime did not retain
them; existing trainer diagnostics remain a separate recomputation workflow.
Missing/no-candidate/verification-failed results stay distinct from valid matches.

Execution adds only a Debug completion binding and references to existing private
result/model/cloud evidence. The loader retains original archive bytes it already
read and verified, without additional file reads or serialization. Explicit capture
privately freezes arrays and copies metadata; it adds no new work to Lean execution.
Snapshots retain input packets and cloud capture IDs separately. These are software
pairing/declarations, not proof of physically synchronized sensor exposures.

Exports include `review.json`, original `executed_recipe.s3l`, binary PLY inputs/
default fit/selected display, native NPY XYZ/normals/RGB/pixel mappings, sampled
scene and each accepted pose/support/visibility/free-space mask and scene indices.
PLY preserves units/frame/XYZ/RGB/normals; NPY plus JSON preserves source size,
pixel mapping and metadata. A raw-input selection exports that raw display and
records its name. The assessment PNG records yaw/pitch and is a display projection,
not measured image coordinates. Original captured data and acceptance outputs
are unaffected. Existing folders are never overwritten. Save As copies the
sealed recipe with its original checksum.

Cross-frame comparison, picking/focus/zoom enhancements, retaining per-ray maps,
independent matching accuracy and hands-on operator acceptance remain future work.
