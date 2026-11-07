# Join images

Open [join_images.json](join_images.json) with **File → Open…** and click
**Run once**. The 652 × 360 output shows the original inspection image
on the left and a 160 × 200 filtered crop on the right, centered vertically
and separated by a 12-pixel gap. The layout places the crop at
`[492, 80, 160, 200]` in output `[x, y, width, height]` coordinates.

Find **Join images** under **Image processing → Geometry**. Wire two images
with the same pixel depth, channel count, and color encoding. Horizontal
joins place Image A on the left. Vertical joins place it on top.
**Strict** requires equal heights for horizontal joins and equal widths
for vertical joins. **Pad** accepts different sizes on that axis and
aligns the shorter image at the start, center, or end. **Fill value**
applies in stored units to all channels in the padding and gap.

The join has its own image frame. **Layout** records the output rectangle,
original coordinate context, and any pyramid metadata for each panel.
When both inputs carry tickets, they must have the same ticket identity.
The output is limited to 16,777,216 pixels. [OpenCV concatenation reference](https://docs.opencv.org/5.0/main_modules/core_array.html#hconcat)

```powershell
opencv-dataflow-run "examples/custom tools/join_images/join_images.json"
```

[All examples](../../README.md)
