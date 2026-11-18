# T-LESS model-source comparison data

T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects.
Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis,
and Xenophon Zabulis. WACV 2017.

[Original project, citation and license](https://cmp.felk.cvut.cz/t-less/)
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
[Official BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless)

Pinned mirror revision: `5fd309a04476a842d93abfb584fba9ee7caecdf1`.
`reference_provenance.json` records SHA-256 for the complete selected corpus,
the fully verified model ZIP identity and the upstream test ZIP identity.
Selected test members were acquired by bounded HTTP ranges, checked through
ZIP CRC and local hashes; the entire test archive was not downloaded or hashed.

Unmodified data: CAD and reconstructed meshes for objects 6 and 8, model
metadata, and original RGB/depth/mask/scene metadata. Object 6 teaches from
scene 2/frame 3 and tests 13/24 plus original 255/494. Object 8 teaches from
scene 3/frame 5 and tests 10/16 plus original 261/493. Cross-scene original
captures are target-absent negatives. `reference_selection.json` freezes these
assignments. Nearby frames are the next two available IDs, chosen before
acquisition or matching. Original subset selection/provenance metadata is
retained for context; the reference-prefixed files define this new comparison.

Derived outputs backproject measured first returns, use supplied masks only
for explicitly isolated cases, and transform the teaching capture into model
coordinates using its annotation. Test annotations are evaluation only. No
hidden sensor surfaces are added. Publisher CAD/reconstructed meshes represent
expected geometry, not additional sensor returns. The teaching capture replay
is a control, excluded from test totals. This is development comparison, not
independent production qualification or an official BOP benchmark.

Raw and derived data retain CC BY 4.0; implementation code uses MIT.
Preserve attribution on redistribution. No endorsement is implied.
