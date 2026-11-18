# T-LESS attribution and modifications

**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**, by
Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis and
Xenophon Zabulis, WACV 2017.
[Original project, license and citation](https://cmp.felk.cvut.cz/t-less/) ·
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The primary project's license and citation were checked on **2026-11-18**.

This increment reuses prepared measured clouds, CAD-derived recipes and truth
from official BOP mirror revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`.
The examples use object 29/scene 1 PrimeSense capture 230 and object 6/scene 2
capture 3. [Object-29 attribution/source hashes](sources/object29/ATTRIBUTION.md)
and [object-6 records](sources/object6/ATTRIBUTION.md) retain upstream archive
members, selected raw inputs, masks, camera/depth conversions and limitations.
No new archive or photographs were downloaded for this increment.

Existing preparation converts first-return measured depth into XYZ millimeters,
keeps missing returns absent, uses publisher visible masks for isolated search,
samples CAD normals, and retains the camera/full measured depth context. CAD
reference geometry is distinct from measured scene returns. Ground-truth poses
remain evaluation-only and never seed localization. These selected benchmark
captures are development examples, not independent full-dataset validation.

New derivatives are colored kept/excluded crop previews, an explicitly derived
bounds glyph, native PLY/documentary PNG fit views and the combined gallery.
Retrained archives retain identical reference/sample/CAD arrays and settings.
Crop controls are undone before fitting. The existing accepted pose and missed
localization remain unchanged. Dataset content and derivatives retain CC BY 4.0;
no endorsement is implied. Generated bounds graphics and software checks do not
establish physical sensor/pose accuracy.

Authored code and the bundled tool retain the included [MIT license](LICENSE).
OpenCV surface-matching dependency notices remain separate from project and
dataset licenses.
