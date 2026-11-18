# Image sources and licenses

## Real weld photographs and supplied masks

Hyun Soo Kim (2024), *Weld bead images and masks*, Figshare version 1.
[Dataset/DOI](https://doi.org/10.6084/m9.figshare.25904014.v1) ·
[Primary metadata](https://api.figshare.com/v2/articles/25904014/versions/1) ·
[Associated publication](https://doi.org/10.5762/KAIS.2024.25.8.184).

Primary version-1 metadata checked on **2026-11-17**: license **MIT**, author
**Hyun Soo Kim**. The complete upstream copyright/license text is retained in
[WELD-LICENSE-data.txt](WELD-LICENSE-data.txt).

Original archive file ID: `46550428`. Reused prepared public images and matching
`_mask.png` files under `weld/assets` retain upstream filenames and bytes:

- `20230930_21h03m42s_B11_1.png`: training reference and first replay.
- `20231113_19h46m30s_B13_3.png`: separate replay.
- `20231114_11h25m52s_B04_3.png`: failing replay.

The saved `weld.spw` contains the pre-existing grayscale training reference.
This increment does not retrain/tune it. Supplied dataset masks remain available
as annotations; the current station reviews do not assert dimensional accuracy
against those masks. The failing frame can include weak/unrelated measured edges.

Changes for review: grayscale inspection input, colored station/scan/edge
overlays, display-only crops, native intensity/gradient profiles and pixel/physical
trends, diagnostic tables and completed-packet metadata. Exact prepared consumed
arrays and original manifest bytes are exported with hashes. Dataset authors do
not endorse this tool.

## Generated controls

The `generated` and `calibrated` assets reuse authored bead/path controls and
planar calibration. They are generated, not acquired-camera data. They demonstrate
missing/ambiguous/narrow/off-path stations and physical versus pixel units.
No real measurement qualification is claimed. Authored code, controls and the
bundled package retain the supplied [MIT license](LICENSE). No private source
repository connection is required to run the examples.
