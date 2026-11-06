# Packed remap coordinates

Open [packed_remap.json](packed_remap.json) with **File → Open…** and
click **Run once**. Two branches straighten the same distorted grid
using floating-point and packed maps. The final block measures the
mean absolute pixel difference. Both branches use the same rectified
coordinate frame and retain the source ticket.

Find **Pack remap coordinates** and **Remap with packed coordinates**
under **Image processing → Geometry**. Feed the packer matching
float32 X and Y coordinate planes, such as those from **Build lens
correction maps**. Wire its packed output to any branch that uses the
same map. **Subpixel** stores six bytes per destination pixel and
supports Linear, Cubic, and Lanczos4 interpolation. Its fractions have
a 1/32-pixel step, so resampled values can differ from the floating-map
path. **Nearest** stores four bytes per pixel and requires Nearest
interpolation. The original float32 pair stores eight bytes per pixel.
[OpenCV map conversion reference](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html#convertmaps)

The packer rejects coordinates outside the representable fixed-point
range. The packed packet is used on graph connections during execution.
Keep the source maps or calibration profile for saving and reloading.

```powershell
opencv-dataflow-run examples/calibration/packed_remap/packed_remap.json
```

[All examples](../../README.md)
