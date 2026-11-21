# Gray-World White Balance

Open the [8-bit project](gray_world_uint8.json) or [16-bit project](gray_world_uint16.json) and click **Run once**. The chart has four neutral patches on the left with a simulated warm cast and four saturated color patches on the right. Compare the results from all pixels, a saturation cutoff of 0.60, and a rectangle over the first neutral patch.

| Estimation source | Informative pixels | Second neutral patch, 8-bit BGR | Second neutral patch, 16-bit BGR |
| --- | ---: | --- | --- |
| All pixels | 14,144 | 48, 43, 44 | 12,336, 11,155, 11,365 |
| Saturation cutoff 0.60 | 5,824 | 48, 48, 48 | 12,336, 12,336, 12,336 |
| First neutral patch | 1,456 | 48, 48, 48 | 12,336, 12,336, 12,336 |

## Use the block

- Accepts unsigned 8-bit and 16-bit RGB, BGR, RGBA, and BGRA. The corrected image keeps the input grid, depth, channel order, ticket, and alpha samples.
- **Saturation threshold** is the maximum difference between the brightest and darkest channels divided by the brightest channel. OpenCV quantizes this cutoff at the input depth. A threshold of 1 admits every color. When available, use a mask or shape region around neutral material to improve the estimate. When both are connected, their intersection supplies the candidate pixels.
- The block estimates gains from the admitted pixels and applies them to the whole input. Gains are normalized to a maximum of 1, so the correction attenuates stronger channels and can darken the image. The **Estimation support** preview marks the admitted nonblack samples. **Report** includes the number of usable pixels and **Gains** reports the applied fixed-point gains.
- An estimate needs signal in all three channels. If the selection or cutoff leaves insufficient signal, the block reports a failed estimate and withholds the corrected image.

[OpenCV GrayworldWB reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xphoto_1_1GrayworldWB.html)

The original synthetic charts are covered by the repository's MIT license. Rebuild the PNGs and projects with `python build_example.py --output <folder>` in the application environment.
