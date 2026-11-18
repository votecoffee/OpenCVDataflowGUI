# Sources, licenses and limitations

**zeng song**, *WLI-Set: A Comprehensive Laser Image Dataset for Real-Time
Measurement of Wheelset Geometric Parameters*, version 3, Figshare,
DOI [10.6084/m9.figshare.25040747.v3](https://doi.org/10.6084/m9.figshare.25040747.v3).

The [versioned publisher API](https://api.figshare.com/v2/articles/25040747/versions/3)
declares [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Author/license metadata were checked directly on 2026-11-18; selected fields
are retained in [publisher-license.json](sources/publisher-license.json).
Preserve attribution and license when redistributing images and derivatives.
No endorsement is implied. Code/generated controls retain [MIT](LICENSE).

This subset copies the original selected first WLI photograph byte-for-byte,
plus its existing reference recipe and ROI, without painting, rescaling or
exposure/threshold adjustment. [Provenance](sources/provenance.json), the original
[comparison summary](sources/summary.json) and published centerline PNGs are
retained from the earlier four-image selection. Only the first photograph is
duplicated here. Centerlines are separate industrial-software references, not
training labels or independent subpixel/metric gauge truth. Publisher whole-ZIP
MD5 values remain metadata; complete large upstream archives were not locally
rehashed as part of this publication. The real case stays pixel-only and fails
the unchanged availability requirement. No camera/laser calibration is supplied.

Generated controls supply analytic pinhole/laser geometry, two known training
target depths and two separate validation depths, a nominal 50 mm step and
deliberate unavailable samples. The target-pixel definition is explicit in
[observations.json](observations.json). It is not acquired calibration imagery,
sensor noise or physical metrology evidence. Validation points never enter the
plane fit. Training residuals and declared target agreement use the same known
camera, with no automatic acceptance criterion or proof of acquisition independence.

Derivatives include retained masks, fitted calibration metadata, taught recipes,
stripe/availability/metric reports, overlays, profile charts and native PLY.
Original files/models remain unchanged; finished staged recipes are new assets.
Measured profile points contain accepted intersections only; missing/rejected
bins remain unavailable. Native fly-around output and frozen Results do not
invent hidden surfaces, fill gaps or re-estimate a completed calibration.
Independent physical gauges and real-data extraction/stripe-identity checks
remain necessary. No private data or private repository attribution is included.
