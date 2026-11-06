# Linear channel transform

Open [channel_transform.json](channel_transform.json) with **File → Open…**
and click **Run once**. Both branches use the same rectangular region of a
synthetic BGR image.

- **Subtract green from blue** replaces blue with `B − 0.5 × G + 10` and
  retains green, red, and pixels outside the patch.
- **Red minus blue** generates a grayscale image from `R − B + 128`. The
  offset makes zero difference visible as mid-gray. Pixels outside the
  patch are zero because the output changes from color to grayscale.

Find **Linear channel transform** under **Image processing → Color and
depth**. Use one coefficient row for grayscale or three for a color output.
Columns follow the stored color order, with an optional offset in the last
column. A wired **matrix** takes precedence over the configured value.
An empty matrix passes the image through. Mask and region inputs can be
combined. The block retains source depth, clips integer outputs to the
depth range, and copies alpha on color output.
[OpenCV transform reference](https://docs.opencv.org/5.0/main_modules/core_array.html#transform)

```powershell
opencv-dataflow-run examples/image_processing/channel_transform/channel_transform.json
```

[All examples](../../README.md)
