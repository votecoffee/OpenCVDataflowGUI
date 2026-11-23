# Extended Inpainting

Open [extended_inpainting.json](extended_inpainting.json) and click **Run once**. The synthetic surface has known repeating texture and a damaged 8 × 7 rectangle. Three branches fill the same area. Inspect **Image** for the repair and **Repair Mask** for the coordinates of the estimated pixels.

| Input or method | RMS error in selected pixels | Largest selected-channel error |
| --- | ---: | ---: |
| Damaged source | 115.19 | 196 |
| FSR Fast | 1.03 | 2 |
| FSR Best | 1.10 | 2 |
| Shift Map | 0.19 | 1 |

Errors are in 8-bit levels against the [clean synthetic reference](assets/clean.png). All three outputs preserve every measured pixel outside the rectangle.

## Use the block

- Find **Extended inpainting** under **OpenCV → Extended Computational Photography (xphoto) → Filters**. Connect a mask, a region, or both. Nonzero mask pixels are selected. With both connected, the block repairs their intersection. An empty selection copies the input, and a full-image selection reports a validation error because it leaves no known pixels.
- **FSR Fast** and **FSR Best** reconstruct frequency content using different quality profiles. They use 8-bit working samples and convert the estimated pixels back to the source dtype. **Shift Map** uses repeated patches, retaining high-depth samples through its float working buffer. Shift Map needs a crop at least 32 × 32 with at most 480,000 pixels, one fully known 8 × 8 patch, and at most 40,000 selected pixels including the native four-pixel halo.
- The output preserves source encoding, alpha, depth, ticket and coordinate metadata. The repair mask identifies synthesized pixels for downstream inspection. Floating-point intensities use [0, 1].

[OpenCV xphoto inpainting reference](https://docs.opencv.org/5.0/extra_modules/xphoto.html)

The images are original synthetic fixtures covered by this repository's MIT license. Rebuild the PNGs and project with `python build_example.py --output <folder>` in the application environment.
