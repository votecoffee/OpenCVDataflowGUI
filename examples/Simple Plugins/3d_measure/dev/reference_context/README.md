# Retained real source context

Requires Simple3DMeasure **0.1.0.dev4** and the native teaching host API.
Open [project.json](project.json), run it and choose **Train 3D measurements…**.
The retained frame-3 reference opens with the earlier plane definitions.

- Toggle **Show source RGB** to see the source appearance alongside region
  boxes and datum axes. Switch it off for green/red fit evidence.
- **Reference context…** shows source image geometry, original camera record,
  source/measurement frames, supplied fixture and source hashes.
- **Review trained / completed run…** switches between cached fit evidence and
  source measured points. Capture the completed frame-13 input to compare;
  the frozen capture identity stays attached when graph outputs change.
- **Export visible review…** includes the visible PLY, recipe, report/context
  and portable full-source NPZ. Reimport NPZ to retain its RGB/pixel/context.
  Stage prepares a draft; inspector Apply activates it; Cancel preserves active settings.

The [source RGB image](../real_tless/source/test_primesense/000002/rgb/000003.png)
and [source-color cloud preview](reference_source.png) identify the retained
capture. [Fit preview](reference_fit.png) shows the same source in measurement
coordinates with the existing regions and fit graphics. These PNGs are computed
views; the project and popup use native flyaround clouds with complete source rows.

The frame-13 numerical report is exactly equal to the earlier
`real_tless/frame000013/expected.json`: height 48.034523 mm, angle 1.703947 degrees,
surface range 21.051655 mm. The reference definitions, learned fit bands and
5-degree parallel guard are unchanged. There are no acceptance limits or
independent dimensional gauge truth. RGB/context does not change calculations.

The `.s3m` contains frame-3 RGB and original pixel rows/size, bounded camera/
dataset metadata and the reference-to-camera transform. The scene PLY preserves
frame-13 XYZ/RGB; this host file format does not preserve arbitrary calibration
or original pixel rows. Native upstream cloud captures and portable NPZ support
that context. Execution snapshot identifiers are not camera exposure timestamps.
See [context definitions and limits](../REFERENCE_CONTEXT.md).

Dataset: T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
*T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*, WACV 2017.
[Official source/license](https://cmp.felk.cvut.cz/t-less/) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ·
[Attribution and pinned raw files](../real_tless/ATTRIBUTION.md).
All derived dataset material retains CC BY 4.0. The fixture is supplied ground
truth, not a detected pose. No dataset endorsement is implied.

Rebuild into a fresh folder with plugin/host installed:

```powershell
python build_reference_context.py real_tless/source new_reference_context
```
