# Recorded depth conversion audit

This numerical audit checks the three retained T-LESS frames (3, 13 and 255)
against their source depth PNG and per-image camera records. The
[official BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)
defines `depth_scale` as the multiplier into millimeters and supplies intrinsics
per image. The audit uses those records, including the retained crop intrinsics.

All three stored meter arrays exactly equal the declared conversion rounded to
float32; missing source pixels remain missing. Maximum depth rounding is
`1.19e-7 m`. Frame 255's full measured cloud has 382,539 points, each mapping to a
unique valid source ray. Maximum XYZ rounding is `0.000098 mm`; maximum pixel
reprojection rounding is `0.000033 px`. These are representation checks with
float32 bounds, not adjusted inspection thresholds.

`audit.json` pins the audited arrays and measured cloud. Reproduce from the parent
example folder:

```powershell
python audit_depth_conversion.py . C:\path\to\new-audit
```

This verifies conversion against supplied declarations. It does not validate
physical camera calibration, annotation alignment, CAD accuracy or sensor bias,
and it does not explain the retained CAD/depth disagreement. The
[independent depth renderer comparison](../depth_diagnostic/README.md) still
rejects the pose-assisted cases under the original 3 mm tolerance. No pose or
clearance acceptance gate changed.

T-LESS source and derived arrays/results are **CC BY 4.0**. Preserve
[attribution](../tless/ATTRIBUTION.md) and [source pins](../locator_chain/source/provenance.json).
T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
*T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*,
WACV 2017. [Publisher](https://cmp.felk.cvut.cz/t-less/),
[license](https://creativecommons.org/licenses/by/4.0/). Audit code is MIT;
no dataset-author endorsement is implied.
