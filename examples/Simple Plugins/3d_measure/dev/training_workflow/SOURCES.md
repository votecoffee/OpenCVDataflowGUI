# Sources and licenses

Real depth-derived reference/clouds, supplied fixture annotations and derivative
models/results use **T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less
Objects**, by Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis
Lourakis and Xenophon Zabulis, WACV 2017. The publisher declares
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
on the [original project page](https://cmp.felk.cvut.cz/t-less/), checked 2026-11-18.
No endorsement is implied.

Scene 2 PrimeSense frame 3 supplies the retained training reference; frames 13
and 24 are separate first-return captures. The original selected-member hashes
and processing are in [provenance](sources/provenance.json) and
[attribution](sources/ATTRIBUTION.md). The raw upstream archive has not been
independently rehashed. This subset reuses existing calibrated XYZ, model bytes
and annotation fixtures, changes only relative project asset paths, and adds
training workflow checks, retraining receipts and native colored PLY/PNG results.
Backsides, hidden CAD surfaces and artificial observation points are not added
to the measured clouds. Graphic grids, boxes, axes and dimensions are derived.

The provided fixtures are ground-truth annotations, not a localization result.
No independent dimensional gauge truth or acceptance limit is available for
these examples. Nonplanar support, residual spread and extrapolation remain
visible. New training does not tune regions, thresholds or algorithms to make
these cases pass.

Generated plane/missing-surface controls and code use [MIT](LICENSE). The control
has a retained 5 mm step reference and an independent missing-target scene.
It demonstrates unavailable measurements and known finite-point geometry;
it is not an acquired sensor-accuracy test. The wheel carries its MIT notice.
