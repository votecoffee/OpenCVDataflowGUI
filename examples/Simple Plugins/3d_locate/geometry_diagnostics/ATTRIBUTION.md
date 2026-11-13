# T-LESS diagnostic data attribution

T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects, Tomas Hodan, Pavel Haluza, Stepan Obdrzalek, Jiri Matas, Manolis Lourakis, and Xenophon Zabulis, WACV 2017.

[Original project, citation and license](https://cmp.felk.cvut.cz/t-less/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Official BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless).

Source revision: `5fd309a04476a842d93abfb584fba9ee7caecdf1`. Object 6 uses scene 2 captures 3/255/494; object 8 uses scene 3 captures 5/261/493. Source member hashes are in source_hashes.json.

Modifications: backproject measured depth in millimeters, apply supplied visible masks, sample CAD, color evidence categories at annotated poses, and draw secondary photographic overlays. No scene returns are invented. All poses here are supplied annotations, never detections.

Raw data and these derived clouds/photographs retain CC BY 4.0; the diagnostic implementation is MIT. No endorsement is implied.
