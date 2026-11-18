# Sources, license and limitations

**zeng song**, *WLI-Set: A Comprehensive Laser Image Dataset for Real-Time
Measurement of Wheelset Geometric Parameters*, version 3, Figshare.
DOI [10.6084/m9.figshare.25040747.v3](https://doi.org/10.6084/m9.figshare.25040747.v3).

The [versioned publisher record](https://api.figshare.com/v2/articles/25040747/versions/3)
declares [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The live publisher API license and author metadata were rechecked 2026-11-17.
Preserve this attribution and license when redistributing images and derivatives;
no endorsement is implied. Dataset terms are separate from this example's MIT code.

The four original JPEGs and corresponding published binary centerline PNGs are the
first, one-third, two-thirds and last sorted original basename, selected before
running extraction. [Provenance](sources/provenance.json) retains publisher IDs,
member sizes/CRC32, local SHA-256 hashes and nested-archive identity. Original JPEG
bytes are stored under content-hash names in `projects/assets`; centerline PNGs are
stored unchanged in `sources/centerline`. Originals and reference labels are not
painted, cropped or rescaled. The source publisher whole-ZIP MD5 values remain
publisher metadata, not claims that complete large archives were locally rehashed.

Changes/derivatives: a separately taught ROI, retained reference recipes, stripe
reports, diagnostic overlays, source/trend selection markers and assessment PNGs.
Original source and reference pixel bytes remain unchanged. No per-test-image
threshold relearning or parameter search was performed. Real failed/input-ceiling/
clipped/ambiguous cases remain failures. The existing
[reference-comparison summary](sources/summary.json) is preserved, with its limits:
distance to any industrial-software centerline is not physical/subpixel gauge
accuracy, sheet identity, completeness or independent dimensional ground truth.
The selected real images include no camera/laser-plane calibration and never produce
metric clouds.

Generated controls are original MIT example data: known pinhole/laser geometry and
an independent nominal 50-mm step, with injected unavailable samples. Additional
negative controls replace the input by zeros or deliberately use an incorrect
laser-plane definition that intersects behind the camera. The latter is explicitly
bad calibration, not a calibration result. Generated geometry is software evidence,
not real sensor noise, physical calibration or gauge repeatability/bias evidence.

Native clouds contain only recorded accepted calibrated profile points. Optional
origin-to-sample segments, configured plane and camera axes are derived display
geometry, not additional sensor returns or newly intersected rays. Missing/rejected
bins stay unavailable. Reviews use exact completed input/model evidence and never
run extraction, undistortion, intersection or calibration. Geometric calibration
declarations and software tickets do not establish physical acquisition truth.

The wheel and example code preserve the original MIT copyright in [LICENSE](LICENSE).
No private dataset or private repository attribution is included.
