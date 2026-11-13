# Frozen novel-model T-LESS validation captures

Dataset: **T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**.
Authors: Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas,
Manolis Lourakis, and Xenophon Zabulis. WACV 2017.

- [Original project, paper and license](https://cmp.felk.cvut.cz/t-less/)
- [Official BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The original project declares **Creative Commons Attribution 4.0 International**.
The mirror's pinned revision is `5fd309a04476a842d93abfb584fba9ee7caecdf1`.
Raw assets and derived clouds, recipes, reports and overlays retain CC BY 4.0,
separately from the implementation's MIT license. Preserve attribution when
redistributing them. No endorsement is implied.

Unmodified upstream members include CAD objects 6 and 8, CAD model metadata,
scene 2/3 metadata and six measured PrimeSense RGB/depth/visible-mask captures:

| Target | Positive scene | Capture IDs | Target-absent negatives |
| --- | --- | --- | --- |
| 6 | 2 | 3, 255, 494 | Scene 3: 5, 261, 493 |
| 8 | 3 | 5, 261, 493 | Scene 2: 3, 255, 494 |

Selection used metadata before matching: the first object ID unique to each
scene, excluding continuous symmetries, and first/middle/last available BOP19
depth filenames. [selection.json](selection.json) records the frozen rule and
settings. Each model has three supplied-mask isolated positives, three full
positives and three target-absent full negatives. All measured positive depth
pixels are retained in full cases; no application workspace or annotated box
is used. CAD alone trains each recipe; test poses/symmetries are evaluation-only.

[provenance.json](provenance.json) records hashes of selected raw members and
the selection manifest. The model ZIP hash was verified locally. The test ZIP
hash is its upstream LFS identity; only the selected members were downloaded
using bounded HTTP ranges and checked through ZIP CRC plus local SHA-256.
The entire test archive was not downloaded or rehashed.

Derived outputs convert measured depth to XYZ in millimeters, apply supplied
masks for isolated cases, train bounded CAD samples, retain camera/full-frame
depth context, and draw predicted pose/fit overlays. Missing returns stay
absent. No hidden scene points are invented. The training-budget limit failed
before matching on this subset; the generic budget fix was then evaluated on
it. This is a development validation set, with failures retained, rather than
an independent production benchmark.
