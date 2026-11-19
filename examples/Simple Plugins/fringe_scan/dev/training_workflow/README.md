# Teach a FringeScan reference

These workflows use FringeScan dev4 and reuse existing example images, recipe
archives and capture manifests. In OpenCVDataflowGUI, open the **FringeScan**
example collection and choose a project under **inspection**.

| Project | Expected result |
| --- | --- |
| `analytic_clean.json` | Generated calibrated control. 6,912 of 6,912 metric pixels accepted. Native cloud. |
| `connected_missing.json` | A clean retained reference and a separate generated capture with missing observations. 6,415 of 6,912 metric pixels accepted. Fails the 95% availability gate. Native cloud keeps the gaps. |
| `warsaw_crown_phase.json` | Attributed external simulation with six phase shifts. 56,985 of 559,872 phase pixels accepted. Fails the 95% availability gate. Wrapped phase image; metric XYZ unavailable. |

1. Select the FringeScan block, then open **Set up sequence and calibration…**
   in Configuration. The retained observation names, source
   images, inspection region and calibration restore. The setup summary
   shows camera/projector frames, baseline, units, exposure and coding scope.
2. Select a row to see the corresponding source image. Gray rows list axis,
   bit and polarity. Phase rows show the declared shift in radians.
   **Audit source sequence** counts available, nonfinite and ceiling samples
   by frame inside the region. Turn on **Show unavailable observations on
   source** for magenta nonfinite and red ceiling pixels. The audit counts input
   availability. The decode and geometry checks run during Preview.
3. **Preview reference** to inspect modulation, residuals, decoding reasons,
   geometry and coverage. Use **Preview another capture…** or **Preview connected
   capture** for a separate inspection. The reference health stays visible as
   the preview changes. With metric points, **Open native result cloud…** starts
   the colored fly-around. Phase-only or empty results retain their image maps.
4. Use **Stage recipe**, then **Apply configuration** in the inspector. Editing
   a field or region clears the prior audit and previews. Correct invalid
   settings and preview the edited reference again. Cancel preserves the
   applied recipe.

The two analytic captures are software controls. The Warsaw example uses
Michał Własiuk and Robert Sitnik’s *Simulated structured light 3D scanner datasets*
(v1.0.0, CC BY 4.0). The source record and transformations are retained with the
completed-result bundle. Acquired-camera and physical calibration qualification
remain open. [Dataset record](https://zenodo.org/records/17826191) ·
[Source hashes and transformations](../completed_runtime/SOURCES.md)

## Reproduce the checks

Run the checker with the host interpreter. Use a new output folder. In this
source repository, point `--baseline` at `FringeScan/examples/completed_runtime`
and use a trusted dev4 plugin loader when the host has an older version. In the
relocated host checkout, `--baseline` points to `fringe_scan/dev/completed_runtime`
and `--projects` points to `fringe_scan/inspection`. The host’s pinned dev4 plugin
can load automatically.

```powershell
python check_training.py --baseline ../completed_runtime --projects ../../inspection --output C:\scratch\fringe-training-check
```

The checker uses the actual inspector offscreen. It exercises restore and
Cancel, invalid input, linked audit, region edit and restore, preview, native
cloud, Stage and Apply, cached Results and exact recipe Save As. The output
contains newly staged projects and diagnostic images. Add `--verify-previous`
with the original, unsplit baseline to replay all eight historical cases.
That replay checks reports, maps, graphics and XYZ/RGB against the original pins.
The sequence audit is session-only. Original inputs and evidence stay in the
existing completed-result and inspection collections.
