# Blend with weight maps

Open [blend_weight_maps.json](blend_weight_maps.json) with **File → Open…**
and click **Run once**. The left portion of the inspection target stays
sharp and the right portion uses a blurred copy. **Soften the boundary**
blurs the rectangle mask so the result transitions gradually between
the two images.

Find **Blend with weight maps** under **Image processing → Image math**.
Connect two images with matching dimensions, depth and channel encoding.
Grayscale and RGB/BGR-family images support uint8, uint16, int16 and
float32. Integer weight maps use uint8 or uint16 full scale. Float weight
maps use values between zero and one. The blend preserves the input depth.

With just **Weight A**, Image B gets the complement. With two weight
maps, each pixel uses their relative weights. **Zero weight** controls
whether to keep Image A or write zero when both weights are zero.
Optional masks and regions preserve Image A outside their intersection.
Native blending uses float32 weight precision. [OpenCV blend reference](https://docs.opencv.org/5.0/main_modules/namespace_cv.html#blendlinear)

**Same frame** checks the known spatial contexts of both images and
weight maps. **Pixel grid** pairs them by index across different frames.
Ticketed inputs must share a ticket. The output uses Image A's coordinates.

```powershell
opencv-dataflow-run examples/blend_weight_maps/blend_weight_maps.json
```

[All examples](../README.md)
