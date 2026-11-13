# T-LESS selected real RGB-D captures

Dataset: **T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**.
Authors: Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas,
Manolis Lourakis, and Xenophon Zabulis. WACV 2017.

- [Original project, paper and license declaration](https://cmp.felk.cvut.cz/t-less/)
- [Official BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless)
- [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/)

The original project and official mirror identify the dataset license as
**CC BY 4.0**. These assets retain that license, independently of the code's MIT
license. Preserve this attribution and license link when redistributing them
or derivatives. No endorsement by the dataset authors is implied.

The selected unmodified upstream files are the CAD mesh of object 29, CAD model
metadata, scene 1 metadata, and real PrimeSense RGB/depth/visible-mask captures
220, 230, 347, and 488. They were extracted from `tless_models.zip` and
`tless_test_primesense_bop19.zip` at mirror revision
`5fd309a04476a842d93abfb584fba9ee7caecdf1`.
[provenance.json](provenance.json) lists SHA-256 hashes for every selected file.
The model archive's SHA-256 is
`6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928`.

`../build_tless_demo.py` produces derivatives: depth-to-XYZ conversion in
millimeters, sampled CAD normals, portable trained recipes, point clouds and
pose/fit overlays. In isolated cases, the supplied visible mask selects object
pixels. In full-scene cases, one fixed 300–1100 mm depth workspace is used.
Missing depth pixels remain absent. No backs, hidden surfaces, or invented
points are inserted into measured scenes. The original depth captures already
encode first-surface sensing and sensor artifacts.

Ground-truth poses and the model's discrete 180-degree symmetry are used only
to evaluate predictions. Segmentation is supplied by the dataset for isolated
cases; Simple3DLocate does not predict those masks. Full-scene cases use no
object masks. These captures are benchmark test data, so the tiny subset is a
development example, not an independent claim of full-dataset performance.
