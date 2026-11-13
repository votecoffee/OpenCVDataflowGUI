# Simple3DMeasure line examples

Requires **opencv-simple-3d-measure 0.1.0.dev2**. All projects use the native
colored flyaround `fit` cloud, structured `lines`/`measurements`/`report`
outputs and the same retained-reference training editor.

| Generated project | Expected result |
| --- | --- |
| [Independent good sample](generated/good/project.json) | 3.9986 mm separation, 30.0048 degrees, 5.0001 mm probe and 20.0227 mm observed span. |
| [Wrong gap](generated/wrong_gap/project.json) | 4.2986 mm separation fails the 3.9–4.1 mm design limits. |
| [Missing target](generated/missing_target/project.json) | Separation is unavailable; the remaining datum measurements remain available. |
| [Nonparallel target](generated/nonparallel/project.json) | The unchanged 5-degree guard leaves separation unavailable. |
| [Shortened observation](generated/shortened/project.json) | 13.9872 mm observed span fails the 19.5–20.5 mm design limits. Hidden length is not inferred. |

These are generated numerical controls with independent noise and unrelated
points. They represent observed line traces, not a complete depth-camera
simulation. The generated design limits are fixed before execution. Runtime
does not widen fit bands or change the training recipe.

| Real T-LESS section project | Separation (mm) | Angle (degrees) | Observed datum span (mm) |
| --- | --- | --- | --- |
| [Frame 3](real_sections/frame000003/project.json), training replay control | 50.0042 | 2.4393 | 49.9958 |
| [Frame 13](real_sections/frame000013/project.json), independent capture | 50.0121 | 0.7575 | 107.8000 |
| [Frame 24](real_sections/frame000024/project.json), independent capture | Unavailable | 12.2927 | 15.1910 |

The real projects fit axes to two fixed, 2 mm-wide table-section volumes in
the retained object coordinate frame. They use measured first returns from
the original RGB-D captures and explicitly supplied annotation fixtures.
They do not detect physical edges, estimate a fixture or establish dimensional
accuracy. No mask, CAD samples or hidden surfaces are inserted. One recipe
trains on frame 3 and runs unchanged on frames 13/24; there are no acceptance
limits or independent gauge dimensions for these sections.

Both line fits in frame 24 satisfy the configured point/inlier criteria, but
the datum has only 38 selected points and about 15 mm of observed span. Its
12.29-degree angle exceeds the unchanged separation guard. A high inlier
percentage is not proof of useful angular precision. Learned radial bands
remain about 1.993 and 2.488 mm in all captures. These failures and coverage
differences are deliberately retained.

Open **Train 3D measurements…**, choose a line and **Focus selected** to inspect
its source samples and fitted span. Choose **New line region** to teach another.
**Flip axis** changes its +X direction. Measurement choices are filtered by
primitive type. In **Review trained / completed run…**, turn Scene off and
select a line or the Lines/Measurements layers to inspect cached fit evidence.
The Lines layer is the fitted observed span; yellow lines/arcs show dimensions.
Red datum X is line direction, while plane datum blue Z remains the normal.

Original source bytes remain in [the sibling T-LESS source folder](../real_tless/source/)
with publisher hashes; `real_sections/source_provenance.json` identifies the
selected files. [T-LESS author attribution and CC BY 4.0 terms](../real_tless/ATTRIBUTION.md)
also apply to the derived line data, recipes and graphics. Tomáš Hodaň,
Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis and Xenophon
Zabulis, *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*,
WACV 2017. [Publisher and license](https://cmp.felk.cvut.cz/t-less/).
No endorsement is implied. Generated controls and builder code use MIT.

Rebuild to a fresh folder with the plugin and host installed:

```powershell
python build_lines.py new_lines --source real_tless/source
```

See [line definitions and archive compatibility](../LINES.md).
