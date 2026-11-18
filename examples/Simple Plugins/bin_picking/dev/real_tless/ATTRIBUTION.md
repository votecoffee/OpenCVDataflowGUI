# T-LESS first-return replay data

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis.
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
WACV 2017.

[Publisher, paper and license](https://cmp.felk.cvut.cz/t-less/) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ·
[Official mirror](https://huggingface.co/datasets/bop-benchmark/tless).

Pinned mirror revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`. Original
PrimeSense scene 2 RGB/depth frames 3, 13 and 24 and `scene_camera.json` are
preserved under `source/`, with verified selected-member SHA-256 hashes in
`source/provenance.json`. The full upstream ZIP was not downloaded; its listed
upstream digest is not a claim of local full-archive verification.

Frame 3 is the retained training/reference replay; frames 13/24 are held-out
captures of the same arrangement, not independent object identities or bins.
Published camera/world calibration supplies the bin-to-camera fixture.
Rounded rotation matrices within 1e-5 orthonormality tolerance are projected
onto a proper rotation. No object masks, meshes or object-pose GT are used.

Principal points differ between crops. Input assets translate pixels by
integer offsets to frame-3 intrinsics: (0,0), (-3,-3), (-2,-8), respectively.
New pixels are zero/unknown, without interpolation or filled surfaces.
`normalization.json` records the transformation; original files are unchanged.
Stored depth is scaled by 0.1 mm. The virtual world-aligned workspace spans
[-200,-200,0] to [200,200,150] mm, with a 5 mm cup and the same physical
settings and 256-candidate budget on all frames. This is configured geometry,
not an annotation of real bin walls.

Derived normalized images, measured XYZ, retained reference recipes, result
clouds/PNGs and reports retain CC BY 4.0. Preserve this attribution and identify
these modifications on redistribution. No endorsement is implied.
Builder code uses MIT separately.

All three produce **zero eligible picks** at the fixed defaults. They expose
contact/visibility limitations and do not establish suction quality, bulk-bin
performance or physical robot success. No ground-truth grasp labels are
provided. Independent bin scenes, gripper labels and measured outcomes remain
required for those claims.
