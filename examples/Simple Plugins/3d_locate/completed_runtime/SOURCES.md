# Sources and licenses

Real measured clouds, CAD-derived recipes, truth and derived review exports use
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
by Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis and
Xenophon Zabulis, WACV 2017. The [original project](https://cmp.felk.cvut.cz/t-less/)
declares [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
That primary license declaration was rechecked on 2026-11-17.

The selected files are existing derivatives from official BOP mirror revision
`5fd309a04476a842d93abfb584fba9ee7caecdf1`: CAD object 29 and scene 1 PrimeSense
capture 230, plus CAD object 6 and scene 2 capture 3. Attribution and upstream
hashes are retained separately for [object 29](sources/object29/ATTRIBUTION.md)
and [object 6](sources/object6/ATTRIBUTION.md). Those records describe archive
members, selected raw assets, source masks, camera/depth conversion and limitations.

Original builders convert first-return measured depth to XYZ millimeters, retain
missing returns as absent, apply the publisher's visible masks for isolated search,
sample CAD normals and preserve camera/full-depth context. They do not add hidden
CAD geometry to the measured scene. Ground-truth poses are evaluation-only.
These few benchmark test captures are development examples, not independent
full-dataset accuracy evidence. No endorsement by the authors is implied.

This publication copies trained recipes and source clouds unchanged, adding
completed-run software provenance, explicit accepted/rejected fit selection,
native colored fly-around review, raw arrays and derived PLY/PNG/JSON. The real
miss remains; no training, thresholds, candidates or gates were tuned for review.
The same CC BY 4.0 credit applies to those real-data derivatives.

The generated asymmetric two-instance scene is an MIT-licensed complete/fused
observation control. The empty scene is a deliberately generated unknown-data
control, not measured sensor evidence. Neither claims real first-surface sensing.
Code and bundled wheel retain their original [MIT copyright/license](LICENSE).
The dataset license does not replace notices associated with the OpenCV
surface-matching dependency.
