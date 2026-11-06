# Path intensity profile

Open [path_intensity_profile.json](path_intensity_profile.json) with
**File → Open…** and click **Run once**. The block follows an L-shaped
path over a three-channel gradient. The path is 70 pixels long and the
samples are spaced 10 pixels apart. The inspection region excludes
the first point, leaving seven selected samples out of eight. At the
last point (50, 40), the B, G, and R samples are 100, 120, and 100.

**Path intensity profile** is under **Image processing → Statistics**.
Edit **Vertices** or wire a list of image-space `[x, y]` points into
**Path**. A wired list takes precedence. Enable **Closed** to sample
the last-to-first segment without repeating the starting point. The
block samples at equal arc-length intervals up to **Sample spacing**,
with a limit of 8,192 samples. Open paths always include both endpoints.

The profile contains distance, coordinates, and one series per
channel. Choose nearest or linear interpolation. With linear
interpolation, contributing pixels must all be selected. Locations
outside the image, mask, or region keep their place in the profile
and carry null values. The inspector plot shows each channel, and
**Result graphics** provides the path for overlay on the source image.
[OpenCV remapping reference](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html)

```powershell
opencv-dataflow-run examples/inspection/path_intensity_profile/path_intensity_profile.json
```

[All examples](../../README.md)
