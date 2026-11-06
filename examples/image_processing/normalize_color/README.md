# Color normalization

Open [normalize_color.json](normalize_color.json) with **File → Open…**
and click **Run once**. The two **Normalize intensity** blocks process the
same rectangular region of a BGR image.

- **Shared channel scale** maps the combined range to 0–255. The selected
  blue, green, and red samples occupy different parts of that range.
- **Independent channel scales** map each channel's range to 0–255. Each
  channel reaches both ends of the output range inside the selected patch.

Both branches preserve the source pixel depth and leave pixels outside the
patch unchanged. The black and white squares outside the patch show which
pixels were excluded from normalization.

The block is under **Image processing → Contrast**. Connect a mask or region
to limit the samples used for normalization. **Outside selection** controls
whether the remaining pixels are zeroed or retained. For RGBA and BGRA
images, **Preserve alpha** keeps opacity out of the normalization and adjusts
it to the output pixel depth if needed.
[OpenCV normalize reference](https://docs.opencv.org/5.0/main_modules/core_array.html#normalize)

```powershell
opencv-dataflow-run examples/image_processing/normalize_color/normalize_color.json
```

[All examples](../../README.md)
