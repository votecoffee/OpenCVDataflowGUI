# Bayer demosaicing

Open [bayer_demosaicing.json](bayer_demosaicing.json) with **File → Open…**,
then click **Run once**. The same synthetic 16-bit RGGB mosaic feeds
bilinear and edge-aware interpolation side by side. Both outputs recover
the expected RGB values inside the color patches and keep the 16-bit depth.

**Demosaic Bayer image** is under **Image processing → Color and depth**.
Connect a single-channel unsigned 8-bit or 16-bit sensor mosaic. Choose
**RGGB**, **BGGR**, **GRBG**, or **GBRG** from the original sensor tile in
row order. Set **Row offset** and **Column offset** to odd when the first
input row or column starts at an odd sensor coordinate. Leave them even
for an uncropped input or an even offset.

Bilinear and edge-aware interpolation support both depths. **VNG** requires
an 8-bit mosaic with at least eight rows and columns. Grayscale and
opaque-alpha formats require bilinear interpolation. The output keeps the
input size and pixel coordinates. [OpenCV demosaicing reference](https://docs.opencv.org/5.0/main_modules/imgproc_color_conversions.html#demosaicing)

```powershell
opencv-dataflow-run examples/bayer_demosaicing/bayer_demosaicing.json
```

[All examples](../README.md)
