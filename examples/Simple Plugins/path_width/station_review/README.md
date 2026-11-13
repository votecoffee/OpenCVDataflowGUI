# SimplePathWidth station review

Requires **SimplePathWidth 0.1.0.dev12**. These views use the existing saved
generated, calibrated and real weld recipes without changing their settings.
The builder measures each source frame once, then restores the completed
station records for review. Review resamples the selected intensity profile;
it does not search for edge pairs again.

| Example | Selected assessment |
| --- | --- |
| Generated clean bead | [Measured](samples/generated_0_measured/assessment.png) |
| Generated failing bead | [Width violation](samples/generated_1_out_of_tolerance/assessment.png), [missing pair](samples/generated_1_missing/assessment.png), [ambiguous](samples/generated_1_ambiguous/assessment.png) |
| Calibrated clean bead | [Measured](samples/calibrated_bead_0_measured/assessment.png) |
| Calibrated narrow bead | [Physical width violation](samples/calibrated_bead_1_out_of_tolerance/assessment.png) |
| Calibrated offset bead | [Physical offset violation](samples/calibrated_bead_2_out_of_tolerance/assessment.png) |
| Real weld training photo | [Measured](samples/weld_bead_0_measured/assessment.png) |
| Separate real weld photo | [Measured](samples/weld_bead_1_measured/assessment.png) |
| Failing real weld photo | [Width violation](samples/weld_bead_2_out_of_tolerance/assessment.png), [missing pair](samples/weld_bead_2_missing/assessment.png) |

The selected station is the middle of the longest consecutive run for its
status. This is a presentation choice, not a training or acceptance change.
These are saved-recipe diagnostics, not independently measured dimensional
ground truth. In particular, the failing weld frame's weak or unrelated edges
remain visible; the gallery does not claim that its detected edges are correct.

Each folder has a full input-grid **overlay.png**, **profile.png**, selected
pixel trend, optional physical trend, **assessment.png**, and **review.json**.
The assessment crop is resized for display; its pixel bounds are recorded.
Cyan marks the selected scan and measured edges; green indicates passing
measurements, red violations/missing pairs, amber ambiguity and gray
unobserved stations. Missing values remain null, not zero.

The values/limits table keeps pixel and calibrated units separate.
The complete-path verdict also considers stations outside the selected crop.
Only missing pairs count as missing-edge gaps; ambiguous or unobserved scans
remain distinct. Gradients use native input intensity per image pixel.

Open a source project, select **Train path width…**, then **Preview on reference**.
Click the image or either trend to select a station. Use the problem filter
and **Previous/Next**, scroll zoom, **Focus station**, **Fit image** and **100%**.
**Export station…** creates a new review folder without staging the recipe.
The trainer previews its current reference, not the last runtime image.
Direct completed-runtime review remains follow-up work.

[samples/receipt.json](samples/receipt.json) records source projects, frame
positions, recipe/input hashes and complete numeric-output hashes.
Rebuild from the host repository root into a **new** directory:

```powershell
python "examples/Simple Plugins/path_width/station_review/build_views.py" .local/new-path-width-review
```

## Attribution

Generated geometry and builder code use this repository's MIT license.
Real photographs and their derived review graphics are from Hyun Soo Kim
(2024), *Weld bead images and masks*, Figshare v1, under the **MIT License**.
[Dataset](https://doi.org/10.6084/m9.figshare.25904014.v1) ·
[source details](../weld_bead/SOURCES.md) ·
[retained copyright and MIT terms](../weld_bead/LICENSE-data.txt).
Changes here add grayscale overlays, selected crops, trends and annotations.
