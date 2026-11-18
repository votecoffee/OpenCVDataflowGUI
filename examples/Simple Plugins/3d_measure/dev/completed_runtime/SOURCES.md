# Sources and licenses

Real depth-derived clouds, retained references, supplied fixtures and derived
review exports use **T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less
Objects**, by Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis
Lourakis and Xenophon Zabulis, WACV 2017.
[Original project and license](https://cmp.felk.cvut.cz/t-less/) ·
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The primary license declaration was rechecked on 2026-11-17.

The existing example uses scene 2 PrimeSense captures 3/13/24 at official mirror
revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`. Frame 3 teaches the reference;
frames 13 and 24 are separate measured first-return captures used here. Preserve
the [original attribution and transformations](sources/ATTRIBUTION.md) and
[selected upstream member hashes](sources/provenance.json). The original raw
RGB/depth images are not duplicated in this subset. The complete upstream test
ZIP was not independently downloaded or rehashed.

Original builders backproject measured depth to XYZ millimeters and map regions
with supplied annotation fixtures. No object mask, CAD points or hidden backsides
are added to the measured cloud. The fixed reference regions and noise-learned
physical fit bands are reused. Real nonplanarity/noise and extrapolation remain;
there is no independent dimensional gauge truth or acceptance limit. Supplied
fixture annotations are not a localization result. No endorsement is implied.

This publication copies model/cloud bytes unchanged, changes only relative paths
to share identical assets, and adds completed-run provenance, selection/layers,
native fly-around views, physical histograms and raw PLY/NPY/NPZ/JSON/PNG exports.
The same CC BY 4.0 credit applies to real-data trained models and derived results.

Generated plane/line/flatness controls and builder code use [MIT](LICENSE).
They retain known finite-point geometry, noise and missing-support definitions;
they are not physical accuracy evidence. The graphics-off control reuses real
frame 13 with the exact model, changing rendering flags only. No retraining,
application-specific thresholds, ROIs or fitting choices were used to improve
these completed review examples. The bundled wheel preserves its MIT notice.
