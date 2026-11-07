# Measure image norms

Open [image_norm.json](image_norm.json) with **File → Open…** and click
**Run once**. The graph adds seven sample levels to each color channel in
a 20 × 10 patch of a synthetic image. Two **Measure image norm** blocks
compare that patch with the same region in the original image.

- **Absolute L1 difference** sums the changes across 200 pixels and three
  channels, giving 4,200.
- **Relative L1 difference** divides by the reference patch's L1 norm,
  giving 0.25. It passes at a maximum of 0.25 and fails if the limit is lower.

Find the block under **Image processing → Statistics**. With only **Image**
wired, the block measures that image's norm. A wired **Reference** enables
absolute or relative difference on matching images. A mask or region restricts
the samples. Relative mode requires a nonzero reference norm within the
selection. Hamming and Hamming2 count differing bits or two-bit groups in
uint8 samples. See [OpenCV array operations](https://docs.opencv.org/5.0/main_modules/core_array.html).

```powershell
opencv-dataflow-run "examples/opencv tools/image_norm/image_norm.json"
```

[All examples](../../README.md)
