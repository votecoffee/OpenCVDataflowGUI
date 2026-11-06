# Lens map displacement

Open [lens_map_displacement.json](lens_map_displacement.json) with
**File → Open…** and click **Run once**. The example builds maps from a
synthetic pinhole profile and draws arrows over a color heatmap. The
correction is strongest near the corners of this example.

Find **Inspect lens map displacement** under **Calibration → Lens distortion**.
Connect X and Y from **Build lens correction maps**. The block compares
each sample coordinate with a scale-only reference that uses the source
and output dimensions. The float32 **Magnitude** is the length of this
difference in source pixels. It includes effects from output intrinsics
and rectification when those settings change.

**Summary** reports minimum, mean, peak and RMS displacement over all
output pixels. **Valid mask** marks sample centers inside the source
pixel bounds, and **Valid fraction** reports the covered fraction of the
output grid. The mask measures coordinate coverage. Interpolation and
border settings in Remap determine how neighboring pixels are sampled.

**Heatmap max** sets the top of the palette in source pixels. Set it
to zero to fit the data or use the same value to compare profiles.
**Arrow spacing** enables sampled directions and **Arrow scale** changes
their display length. Arrows start on the output grid and point toward
the sampled source locations expressed on that grid. The example wires
the drawable vectors to **Overlay result graphics** and enables them
in Debug and Lean.

```powershell
opencv-dataflow-run examples/calibration/lens_map_displacement/lens_map_displacement.json
```

[Lens correction maps](../lens_correction_maps/README.md) · [All examples](../../README.md)
