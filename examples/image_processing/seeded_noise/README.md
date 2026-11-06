# Seeded random images and noise

Open [seeded_noise.json](seeded_noise.json) with **File → Open…** and click
**Run once**. **Generate random image** creates a grayscale background with
uniform values from 96 through 160. Two **Add image noise** blocks add
Gaussian noise to the same rectangular patch. Both branches use seed 73,
so **Measure image norm** reports an exact difference of zero. Change one
noise seed and run again to see the comparison fail.

- **Input → Synthetic → Generate random image** creates a grayscale, BGR,
  or RGB image in a chosen size and depth. Uniform integer images sample
  each integer within the configured bounds with equal probability.
- **Image processing → Image math → Add image noise** adds Gaussian or
  uniform noise in stored sample units. A mask or region limits the affected
  pixels. The output keeps the input encoding and depth. Integer results
  round and saturate, and alpha is preserved by default.

Each activation starts a private generator from the configured seed or
optional wired **Seed**. The **Seed** output reports the value used.
Matching seeds, settings, and selections reproduce the same draws on the
installed NumPy version regardless of branch worker order. Use different
seeds for distinct draws. The noise block draws only for selected samples,
so changing the selection can change which draw a pixel receives.
[NumPy random generator reference](https://numpy.org/doc/stable/reference/random/generator.html)

```powershell
opencv-dataflow-run examples/image_processing/seeded_noise/seeded_noise.json
```

[All examples](../../README.md)
