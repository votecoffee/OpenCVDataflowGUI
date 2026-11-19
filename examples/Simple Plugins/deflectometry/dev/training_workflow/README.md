# Teach a Deflectometry reference

Open a project from the Deflectometry **inspection** collection in a
DataflowGUI checkout with the bundled dev4 plugin. The examples reuse the
original capture and recipe assets.

| Project | Expected result |
| --- | --- |
| `analytic_clean.json` | Generated unchanged control. 6,912 of 6,912 comparable pixels and zero over the change limit. Pass. |
| `connected_change.json` | Separate generated test capture. 6,912 comparable pixels, 293 over the retained phase-change limit. Fail. |
| `sofast_self_review.json` | Real SOFAST capture compared with itself. 114,356 of 114,359 comparable pixels. Pass for the configured wrapped-phase gates. Exposure and response calibration are unknown. |

1. Select the Deflectometry block and open **Train reference and inspection
   region…**. The retained filmstrip, limit settings and inspection mask restore.
   Select a frame to see its axis, step and declared shift in radians.
2. Click **Audit X/Y observations** to count complete source pixels for each
   axis and the intersection inside the inspection region. The selected frame
   lists available, nonfinite and at-ceiling counts. **Show most affected frame**
   takes you to the frame with fewest available pixels. To inspect their locations,
   turn on **Show missing and clipped source pixels**. Magenta marks nonfinite
   data and red marks codes at or above the configured ceiling.
3. The setup summary shows the retained period, origin, direction and exposure
   declarations. The audit checks input availability. Preview checks phase fit,
   phase changes and coverage. Wrapped observations can miss whole-period
   changes. Metric geometry requires absolute screen correspondence and known
   camera, screen and reference plane geometry.
4. **Preview reference** reports phase fit coverage for X and Y. Preview a
   separate or connected capture to review the change and availability maps.
   The reference health stays visible. Edits clear the results from that draft,
   and invalid fields show what to correct before Preview.
5. **Stage training** proposes a new retained recipe. **Apply configuration**
   activates it. Cancel preserves the applied recipe. Save As keeps the staged
   recipe and mask byte-for-byte. Source audit counts remain in the session.

The real case retains the OpenCSP Team's SOFAST data, courtesy Sandia National
Laboratories / NSTTF, under CC BY-SA 4.0. Its reduced code values and mask retain
that license and original attribution. A self-comparison demonstrates the
teaching workflow. Distinct acquired captures and independent shape references
are needed for physical qualification. [OpenCSP project](https://opencsp.sandia.gov/) ·
[Publisher data license](https://sandia-csp.app.box.com/s/iftmhdkhgjnmfgefsfmkid011dtv8n3g/file/1341437503062) ·
[Retained originals, hashes and transformations](../completed_runtime/SOURCES.md)

## Reproduce the checks

From this folder in the host checkout, run with the host Python and choose
a new output folder.

```powershell
python check_training.py --baseline ../completed_runtime --projects ../../inspection --output C:\scratch\deflectometer-check
```

The offscreen checker exercises Restore, Cancel, audit, edits, reference and
independent preview, Stage, Apply, and exact recipe Save As and reopen. It writes
audit receipts, the change/availability images and the staged and relocated
projects to the chosen output. The SOFAST derivatives retain CC BY-SA 4.0.

With the original unsplit baseline, omit `--projects` and add
`--verify-previous` to replay all nine prior projects and six frozen exports.
Use `--plugin` to load an explicit trusted dev4 plugin in a fresh process when
the host has an older bundle.
