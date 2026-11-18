# Initial phase comparison applications

Offline scope: three generated registered phase controls and one actual exported
SOFAST self-review. Inputs are complete ordered X/Y image stacks plus capture/
setup/grid/radiometry metadata and an inspection region. Physical screen/camera
reconstruction calibration is not supplied for the real example. Geometry API
checks use independent synthetic known planes and reflection normals separately.

Reuse existing `input.file` and `layout.comment` blocks for the saved native
review baseline. The graphs contain no equipment control, sinks, hooks, plugins
or estimator execution. File-source PASS is loading success; estimator judgments
are retained in the report and summary. No result cloud is created from phase.

Standalone execution uses one cached reference fitter with explicit reference/test
manifests. Retain original arrays, source coordinates and reasons in NPZ, decision/
sequence metadata in JSON and separate fixed-scale review PNGs. Inputs use
linear-response mode where declared, or explicit uncalibrated-code diagnostics
for the real exported example. Common exposure is unknown only in that diagnostic
mode. Whole-period phase ambiguity is always reported.

Acceptance evidence: independent analytic phase truth within 2e-7 rad after
float32 manifest import; correct clean/change/missing judgments at the same
generic settings; exact packaged-vs-source raw-map/reason parity and graphics-off
parity; four save/reopen/run reviews with 64 exact RGB image loads; pinned source
images/license/member hashes. OpenCV 4.12 and host 5 numerical checks cover the
shared fit and separate normal/reflection math.

Remaining evidence: distinct registered real reference/test captures, exposure/
response/screen calibration, physical shape/defect references and repeatability.
No metrology or machine acceptance is claimed. Next deliverable is intuitive
native retained training and a real execution plugin with the same numerical core.
