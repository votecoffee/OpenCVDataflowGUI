# Lens map files

Open [lens_map_files.json](lens_map_files.json) with **File → Open…** and
click **Run once**. The graph loads a bundled map pair and corrects a
synthetic distorted grid. The map file and image use paths relative to
the project folder.

To save your own pair, connect X and Y from **Build lens correction maps**
to **Save lens correction maps** under **Output → Calibration profiles**.
Choose a new `.npz` path and run the export. In a job that reuses the
maps, connect **Load lens correction maps** to **Remap image coordinates**
or **Inspect lens map displacement**. Use the profile and source image
grid for which the maps were built.

The uncompressed archive contains both float32 arrays, the source and
output dimensions, the rectified calibration, and the valid ROI when
available. The loader checks the array shapes and checksums before
emitting the pair. The saver publishes the complete file and preserves
any file already at the destination.

```powershell
opencv-dataflow-run examples/lens_map_files/lens_map_files.json
```

[Lens correction maps](../lens_correction_maps/README.md) · [All examples](../README.md)
