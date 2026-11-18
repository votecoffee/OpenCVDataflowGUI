# BinPicking retained training setup

Load **plugin.py first**, then open a project below. The checksum-pinned dev14
wheel loads explicitly without installing or replacing the host's package.
Restart if another BinPicking version is already imported. A current host with
native cloud teaching, connected captures and independent Results review is required.

| Project | Training and execution evidence |
| --- | --- |
| [Suction control](projects/suction_good.json) | Generated first-return surface; ten eligible alternatives at the existing settings. |
| [Real suction](projects/suction_real13.json) | Real T-LESS depth/reference; conservative zero-pick outcome remains. |
| [Jaw control](projects/jaw_clear.json) | Generated first returns and declared poses; one eligible taught grasp, another unknown. |
| [Real jaw](projects/jaw_real13.json) | Real T-LESS first returns and supplied pose; both taught alternatives remain rejected. |

Finished staged projects also retain the completed training references:
[suction control](projects/suction_good_trained.json),
[real suction](projects/suction_real13_trained.json),
[jaw control](projects/jaw_clear_trained.json), and
[real jaw](projects/jaw_real13_trained.json). Their exact recipe archives are
included. The newly trained jaw projects can preview the saved explicit
pose/depth references without capturing the graph again.

Open the picker's training popup from **Configuration**. Existing depth, camera,
bin, physical settings and any retained CAD/contact frames appear immediately.
The persistent summary identifies declared units/depth scale, camera dimensions,
reference valid/missing depth, bin extent/origin/+Z and coordinate frames. Jaw
controls distinguish internal meters, displayed contact/tool millimeters and
percent evidence controls. TCP mounting never redefines grasp +Z retract.

Suction settings are grouped into **Tool geometry**, **Evidence / limits** and
**Execution / depth**. **Edit bin XYZ bounds…** validates all six boundaries
together; untouched coordinates keep full precision. Cancel keeps the old bin.
The frame stays calibrated upstream. Accepted changes share existing Undo and
require a new reference preview. Floor picks use the original retained depth ray,
independently of float32 native display XYZ. A changed depth scale requires
capture/import again rather than silently rescaling calibrated geometry.

In the jaw trainer, select taught contacts, gripper dimensions and TCP mount as
usual. Bin calibration comes from the supplied base recipe. Legacy recipes can
lack explicit retained pose records: run the recorded source, then choose
**Capture connected reference** to retain the depth and declared part poses.
No pose is inferred from CAD, a cached result or hidden truth. Reopening a newly
staged recipe restores these records without graph inputs.

Both trainers show actionable invalid-setting reasons and disable Preview/Stage
before planning. Edited, unpreviewed and ready states have persistent guidance.
Cached contact/pose/clearance evidence stays frozen and is marked stale after
edits. **Preview reference** evaluates one frozen draft in a background worker;
review every eligible/rejected/unknown candidate in the native cloud. Layer and
fly-around changes do not replan. Scores describe geometry, not pick probability.

**Stage recipe** creates a unique asset and updates only the inspector draft.
Inspector **Apply** activates it. Cancel keeps active settings. Save As carries
the exact staged recipe, including source depth, frame and retained CAD/poses.
The main window follows the screen size and scrolls setup/native content while
preview, Stage and Cancel stay in a fixed two-row footer. Narrow windows can
need horizontal scrolling for the full native toolbar or layer row.

Run in Debug and use **Review completed bin-picking evidence…** on Results for
independent cached source/model/candidate review. Native colored fly-around clouds
are primary; [saved PLY/PNG evidence](evidence) is included. PNGs are point renders,
not desktop screenshots. Cups, jaws, CAD, sweeps and axes are derived graphics;
missing observations remain unknown. Ranked alternatives can share an object:
reobserve after every pick. No robot or ROS execution occurs in these projects.

Run `python check_training.py --output C:\path\to\new-output` using the host
Python. The output directory must be new. The offscreen actual-inspector check
covers restoration, invalid/stale feedback, reversible edits, reference preview,
Stage without Apply, Apply/reopen/Cancel, native Results and exact Save As.
Original-recipe execution is exact. Retaining pose-reference notes creates a
new archive checksum; repeated rigid-transform validation can introduce floating
roundoff in poses. Checks preserve every discrete decision and compare staged
numeric fields within 1e-12 in their declared units, while runtime XYZ/RGB stays
exact. [Qualification](qualification.json) records actual differences.

Real examples have supplied pose/calibration assistance and no grasp-success
labels. Neither real example becomes eligible. Independent real bin scenes,
seal/force/friction tests, full robot reach/collision and physical pick outcomes
remain open, along with hands-on UI acceptance, compact shared controls,
background reference restoration and finer cancellation.
See [sources, license and modifications](SOURCES.md).
