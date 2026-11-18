# Sources, license and scope

Real reference depth/RGB, measured clouds, object CAD, supplied poses and their
derivatives use **T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less
Objects**, by Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis
Lourakis and Xenophon Zabulis, WACV 2017.

The [publisher's original project](https://cmp.felk.cvut.cz/t-less/) declares
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
checked 2026-11-18. Preserve this credit and license on redistribution; no
endorsement is implied. Example code and generated controls are [MIT](LICENSE).

The subset reuses scene 2 PrimeSense references/captures 3 and 13 and object-6
CAD at official mirror revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`.
Original selected member hashes are retained in [jaw provenance](sources/jaw-provenance.json)
and [suction normalization](sources/suction-normalization.json). The complete
upstream archive was not independently downloaded or rehashed. Existing depth
normalization/backprojection, millimeter suction units and metric jaw/CAD
conversion are preserved. We copy source/model bytes unchanged, change relative
project paths, retain explicit reference captures when teaching, and add native
colored PLY/PNG training/results plus software-check receipts. Authored bin/tool
geometry and contact frames remain separate from measured first returns.

Original camera/object-pose annotations provide declared assistance, not a
localization result or ground truth for suction seals, grasp success, robot
reachability or motion. Conservative real zero-pick outcomes remain. No hidden
back surfaces, CAD points or unobserved cavities become measured free space.

Generated controls use analytic first-visible surfaces and explicit finite
geometry/poses. They supply geometry and missing/unknown behavior, not real
sensor noise, force closure, transported-part collision or physical pick truth.
The wheel retains its original MIT copyright. No private data is included.
