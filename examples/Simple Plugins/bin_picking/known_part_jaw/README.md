# CAD-taught jaw alternatives

Eight offline projects exercise **SimpleBinPicking Jaw** and native CAD-contact teaching. They
retain the CAD surface, two authored part-relative grasp frames, calibrated
depth/camera, gripper dimensions and TCP mounting in checksummed `.sbpj` files.
They use the same generic jaw settings. The clear control demonstrates a usable
geometric alternative; this is not a robot motion or physical grasp example.

## Open and run

Load this folder's `plugin.py` explicitly, then open one of these projects:

| Project | Retained outcome |
| --- | --- |
| [Clear first-return box](clear/project.json) | One eligible X-side grasp; the rotated Y alternative extends into unobserved view space and stays unknown. |
| [Observed obstacle](obstacle/project.json) | No eligible grasp. A foreign obstacle intersects the X-side palm sweep. |
| [Unknown clearance](unknown/project.json) | No eligible grasp. Missing rays under the palm remain unknown despite accepted part-pose evidence. |
| [Real T-LESS frame 3](tless/frame000003/project.json) | Both proposals rejected: only 25.10% current model support and 34.93% contradiction. |
| [Real T-LESS frame 13](tless/frame000013/project.json) | Both proposals rejected: 28.96% support and 22.10% contradiction. |

The supplied poses are assistance: analytic ground truth in controls and dataset
pose annotations in T-LESS. The real RGB-D captures are camera-visible first
returns, with their original missing samples. Their annotations do not bypass
the current-depth check. These cases do not measure independent localization,
grasp-label accuracy or physical pick success. Real-data pose verification and
useful CAD contact proposals need further work; thresholds were not relaxed to
make the examples pass.

The plugin loads its bundled `opencv-simple-bin-picking` **0.1.0.dev5** wheel,
without installing or updating the app environment. NumPy and SciPy must already
be available. The complete wheel SHA-256 is
`7cd2702f4775706dfbd7c7fefd40bd4ecf3a40c026c486fd731c81fa9a79a40f`.
Opening a project does not load Python code automatically. If an older
BinPicking package is already imported, restart the app and load this plugin
first; the loader rejects mixed versions. The original suction block remains
available and is not replaced by this plugin.

## Assess the result

Run the graph and select **Cloud / graphics**. Its thumbnail and popout use the
native colored fly-around viewer. The `graphics` output contains:

- Gray measured scene and cyan derived CAD surface.
- Green eligible, red rejected and orange unknown closed-finger/palm frames.
- Blue open fingers and conservative approach/closure/retract envelopes.
- RGB TCP axes, approach segments, magenta collision witnesses and orange
  unknown query points.

`graphics.json` identifies measured versus derived row ranges. Only measured
depth feeds scene clearance; graphics and hidden CAD surfaces never become
observed free space. `expected.json` preserves every candidate and reason,
including raw rejected poses. `tcp` and `pregrasp` are absent if no alternative
is eligible. Scores combine geometric contact and depth support, not probability.

The bin/world frame is Z-up. If the default view shows the floor from below,
set viewer yaw near **155 degrees**, then orbit to assess the jaws. The static
`result.png` uses that orientation; `result.ply` retains actual metric coordinates
and RGB. Initial-view metadata support has been requested from the app owner.

![Clear control review](clear/result.png)

The initial tool checks finite sampled gripper approach, closure and retract.
It permits an explicit bounded CAD contact band and reports that separately
from measured free space. It does not test force closure, transported-part
volume, robot IK, continuous collision freedom or physical grasp success. The
proposals are alternatives, not a preapproved sequence of picks.

## Training and next work

Open **Teach CAD jaw grasps…** from the block's Configuration inspector.
The [native training guide](JAW_TRAINING.md) covers actual-row CAD point picking,
numeric contact/retract edits, named alternatives/undo, gripper/TCP controls,
frozen references, cancellable previews and Stage/inspector Apply/Cancel.
The original model origin and untouched stored values are preserved.

Three projects were trained through the actual inspector and retain their own
raw depth and explicit pose reference, so **Preview reference** works on reopen:

| Native teaching project | Result after preview, Stage, Apply and reopen |
| --- | --- |
| [Clear control](native_training/clear/project.json) | One eligible alternative. |
| [Observed obstacle](native_training/obstacle/project.json) | Zero; collision remains rejected. |
| [Real T-LESS frame 13](native_training/tless_frame000013/project.json) | Zero; annotated pose assistance does not override weak depth support. |

`native_training/checks.json` records actual connected capture, Cancel preserving
applied training, Stage editing only the draft, Apply, and retained-reference
preview after reopen. Existing legacy recipes show CAD/grasps immediately but
need a supplied pose reference once before they can preview independently.
Explicit locator-instance adapters, better general real contact proposals,
symmetry handling and qualified robot handoff remain follow-ups.

## Verification and reproduction

`verification.json` records saved-project reopen and graphics-off/cache parity.
`native-verification.json` records actual offscreen app thumbnail/popout orbit,
RGB preservation, review without replanning and portable Save As for all eight
projects. This is software acceptance, not hardware qualification. The bundled
wheel passed 86 checks against the checkout host, including existing suction
and mocked simulation checks. All 24 packaged source modules and the MIT notice
were checked against the source publication, allowing newline normalization.

From an environment with the app, NumPy, SciPy and OpenCV:

```powershell
python build.py C:\path\to\new-output releases/opencv_simple_bin_picking-0.1.0.dev5-py3-none-any.whl --source tless/source
python build_native_training.py C:\path\to\new-output C:\path\to\new-output\native_training
python verify.py C:\path\to\new-output C:\path\to\new-checks
```

Add `--native` to `verify.py` with Qt available to exercise native preview
widgets. For unattended checks set `QT_QPA_PLATFORM=offscreen`. The scripts
create new output folders and never connect to a simulator or robot.

## License and attribution

Example code and authored controls: [MIT](LICENSE). Real T-LESS source and
derived arrays, recipes and results: **CC BY 4.0**, independent of the code
license. Preserve [T-LESS attribution](tless/ATTRIBUTION.md) and provenance when
redistributing those assets. No endorsement by the dataset authors is implied.
