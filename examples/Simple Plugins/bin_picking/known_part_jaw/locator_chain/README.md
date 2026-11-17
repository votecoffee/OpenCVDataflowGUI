# Real locator → instances → jaw planning

These two projects execute **Simple3DLocate → Simple part instances →
SimpleBinPicking Jaw** on a recorded T-LESS first-return RGB-D capture. They do
not supply annotated object poses. A publisher-provided visible instance mask
supplies the isolated locator scene; this is assisted segmentation, not an
independent instance detector.

Load the parent `plugin.py` first. It loads pinned Simple3DLocate `0.1.0.dev20`
and BinPicking `0.1.0.dev11` wheels without modifying the app environment. Restart
if a different package version is already imported.

| Project | Locator matches | Converted instances | Eligible jaw alternatives |
| --- | --- | --- | --- |
| [Publisher reconstructed reference](reconstructed/project.json) | 1 | 1 | 0: current CAD/depth evidence insufficient. |
| [Publisher CAD reference](cad/project.json) | 0 | 0 | 0: empty accepted instance list. |

Both use the same measured PrimeSense scene 2/frame 255 and the same generic jaw
settings/CAD contact frames. The two existing locator recipes are byte-preserved;
their model representations differ. There is no per-case threshold, pose seed
or runtime lookup of test pose labels. The accepted locator proposal is not a
qualified grasp and does not bypass the jaw tool's independent raw-depth check.

The recorded input source emits registered axial depth in meters, the paired
isolated/full measured clouds in their retained millimeter camera frame, and an
explicit acquisition ID. The locator report is in mm. The converter changes
translation units to m without shifting the original model origin. The jaw
recipe uses the same original T-LESS object-6 frame, with a supplied camera/world
calibration. Camera/world calibration is separate from object localization.

Both cloud metadata records and the explicit locator input declare the same
capture ID. The locator validates them before loading the model or matching and
propagates the declaration into reports, poses and fit clouds. The converter
requires that embedded ID, with legacy binding disabled. Equal caller declarations
do not prove exposure or robot-time synchronization. Bind the actual acquisition
that supplied the locator in a live application.

`capture-upgrade.json` records unchanged input/recipe bytes, locator geometry and
exact jaw outcomes across the metadata upgrade. The separate
[capture controls](../capture_controls/README.md) exercise metadata-only binding
and a deliberately mismatched pair. [Contract](../LOCATOR_CAPTURE.md).

Select the locator's **Cloud / fit** or the jaw tool's **Cloud / graphics** for
native RGB fly-around review. All raw candidates and conversion decisions stay
in `locator-report.json`, `adapter-report.json` and `expected.json`. Review does
not rerun either solver. Source depth, clouds and both sealed recipes are
declared assets, so Save As retains their bytes and local references. No TCP is
emitted on these rejected cases, and neither project executes robot motion.

## Source, changes and attribution

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
WACV 2017. [Publisher and license](https://cmp.felk.cvut.cz/t-less/),
[official BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Source and derived assets retain CC BY 4.0, separately from MIT example code.
Preserve attribution and the license link. No endorsement is implied.
Mirror revision: `5fd309a04476a842d93abfb584fba9ee7caecdf1`.
`source/provenance.json` pins selected original RGB/depth/mask/camera files.
`checks.json` pins the derived input clouds, unchanged locator recipes and
original depth PNG. Source object-pose labels are absent from this runtime data.

Changes: masked/full measured depth backprojection and normals in the retained
clouds; publisher model sampling/locator recipes; depth converted to meters;
authored jaw frames, inspection bounds and result graphics. All hidden CAD
surfaces remain predictions rather than additional sensor returns. No physical
grasp, transported-part, robot-path or continuous collision qualification is made.
