# Retain a new FoundationPose object

Load **plugin.py**, open either `worker_execution/native_setup_*.json` project,
select SimpleFoundationPose, and click **Set up object…** in Configuration.
The popup restores the actual RGB/depth/mask, calibration, settings and original
CAD. Opening it does not start inference or require the worker dependencies.
The bundled version is `0.1.0.dev2`; restart the app if an older plugin is loaded.

1. For a new application, import a PLY/STL mesh and declare its stored-unit to
   meter scale. Import runs in the isolated worker. The canonical CAD/origin
   cloud is separate from the measured reference camera cloud; no alignment or
   symmetry is assumed.
2. Import registered RGB/depth/camera/mask image files, a retained NPZ capture,
   or capture the connected inputs before the untrained block has run. Camera
   JSON declares `matrix`, `image_size: [width,height]`, `frame` and rectified
   pinhole/zero-distortion semantics. Declare stored depth units explicitly.
   An omitted file mask starts blank for selecting one visible instance.
3. Zoom/pan the source image and edit the target mask. Add/remove brush, rectangle,
   circle, polygon, RGB magic wand, imported mask labels and undo are available.
   Drag polygon points, Shift-click to insert, and right-click to remove. Drawn
   regions require **Apply drawn region** or **Discard region** before Stage.
   Imported masks remain exact until edited. RGB, measured-depth display and
   mask filmstrip entries share the original pixel grid.
4. Review precise calibration, source units, original part frame and generic
   execution/fit settings. Setup health requires a nonempty mask with at least
   12 raw depths at or beyond the current renderer's 0.1 m near plane. This is
   consistency feedback, not pose accuracy or grasp qualification.
5. Optionally preview the reference or an independent capture. The editor owns
   a separate worker, progress and cancellation; source/reference data never
   become the test image. Edits and changed connected publications discard
   stale previews. CPU global search can take minutes. File/reference previews
   use the chosen initialization; connected previews follow a wired coarse-pose
   input when present. Imported/coarse hypotheses are explicitly reported.
6. Select execution from connected RGB-D or the retained reference. **Stage object
   setup** writes a new sealed filename/hash into the inspector draft. **Apply
   configuration** commits it; Cancel preserves the applied setup. Source images,
   mask, calibration precision and original training files remain retained.

Use **Runtime on this machine…** to choose an existing trusted isolated Python
and external pinned weights. Saving those preferences changes this machine's
profile immediately and is independent of inspector Apply/Cancel. No dependencies
or weights are installed/downloaded. Keep the app environment unchanged.
See [runtime, output interpretation and licenses](WORKER_WORKFLOW.md).

## Example evidence

The two native-setup projects share a setup recreated through the actual image
file wizard and isolated CAD importer from the attributed T-LESS frame-3 inputs.
Geometry/RGB/mask/calibration match exactly. Initial stored-depth conversion
differs from the historical archive by at most 0.0000000597 m, within float32
rounding; subsequent retained calibration/array saves preserve their values.
The example executions use two prior network proposals and explicitly report
that additional initialization assistance. Both reproduce earlier candidates
exactly and reject; no accepted pose or grasp is emitted.

Two full 252-hypothesis editor previews also reproduce the earlier global
candidates exactly and retain rejection, independent reference images and
Stage/reopen behavior. Actual packaged inspector tests verify Cancel, Stage
without changing the applied block, Apply and reopening all source data.
`native_setup_evidence` retains these checks and the package fingerprint.

Core/worker installation: 37 checks pass, two optional host/Qt modules skip.
Host training/adapter checks: 21 pass against both installed and checkout APIs.
The host process imports no ONNX Runtime, Torch or Trimesh for those workflows.
Original execution projects, development measurements, assets and receipts remain
available. Broad pose accuracy, GPU rendering, tracking, detection, declared
symmetry handling and physical grasping remain unqualified.

Inputs and derived training: T. Hodaň et al., **T-LESS**, WACV 2017,
[publisher](https://cmp.felk.cvut.cz/t-less/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Visible masks are supplied annotation assistance; annotated poses enter evaluation
only. See original `source/provenance.json`, [attribution](../README.md) and
[model/code notices](THIRD_PARTY.md). Weights are external and are not redistributed.
