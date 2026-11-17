# T-LESS real-data attribution

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
IEEE Winter Conference on Applications of Computer Vision (WACV), 2017.

- [Original publisher, citation and license](https://cmp.felk.cvut.cz/t-less/)
- [Official BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The publisher and official mirror both identify **CC BY 4.0**; rechecked
2026-11-17. The dataset license applies to the selected source files and their
derivatives, separately from this example's MIT code. Preserve attribution and
the license link when redistributing them. No endorsement is implied.

Selected unmodified source members come from mirror revision
`5fd309a04476a842d93abfb584fba9ee7caecdf1`: object-6 CAD triangle PLY,
PrimeSense scene-2 camera/pose metadata, and RGB/depth frames 3 and 13.
`provenance.json` contains verified SHA-256 hashes. The complete upstream ZIP
was not downloaded for this increment.

Changes: CAD and depth converted from millimeters to meters; 40,000 deterministic
area-weighted triangle samples with face normals; original CAD frame/origin
preserved. Two axial contact frames, gripper dimensions, inspection workspace
bounds and result graphics are example-authored. `.sbpj`, NPY, colored PLY,
PNG and reports are derivatives. Source camera/world calibration and object pose
annotations remain available alongside them.

Object poses enter these examples as explicit ground-truth assistance, rather
than as locator results or grasp truth. Both measured scenes remain rejected
under unchanged generic pose/clearance settings. There are no dataset grasp,
robot-reachability or physical pick-success labels in these examples.
