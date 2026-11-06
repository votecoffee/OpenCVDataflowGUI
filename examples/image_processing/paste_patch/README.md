# Paste image patch

Open [paste_patch.json](paste_patch.json) with **File → Open…** and click
**Run once**. The graph crops a 70 × 70 patch from the circular part and
selects a circle within it. **Paste image patch** places that selection
at canvas position `[-24, 40]`. Clip mode keeps the visible overlap.
The source rectangle is `[24, 0, 46, 70]` and the destination rectangle
is `[0, 40, 46, 70]`, both in `[x, y, width, height]` order.

Find **Paste image patch** under **Image processing → Geometry**. The
canvas and patch must use the same pixel depth and channel encoding.
Enter or wire whole-pixel **X** and **Y** offsets. **Strict** requires the
whole patch to fit. Optional masks and regions are in patch coordinates.
The canvas is copied before applying the patch. [OpenCV masked copy reference](https://docs.opencv.org/5.0/main_modules/core_array.html#copyto)

**Copied mask** and **Copied pixel count** describe the pixels written,
including samples that happen to equal the canvas. **Placement** also
provides the patch-to-canvas transform. The image and mask retain the
canvas coordinate frame. If both images carry tickets, the tickets must
match. An empty overlap or selection emits the canvas and a failed result.
The canvas is limited to 16,777,216 pixels.

```powershell
opencv-dataflow-run examples/image_processing/paste_patch/paste_patch.json
```

[All examples](../../README.md)
