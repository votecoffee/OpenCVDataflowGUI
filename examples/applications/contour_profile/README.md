# Contour profile tolerance

Open [contour_profile.json](contour_profile.json) with **File → Open…**,
then click **Run once**. The project compares a nominal outline with
a sample that has a 6-pixel notch and a 9-pixel protrusion. The maximum
deviation is 9 pixels, which passes the saved tolerance of 10 pixels.
Lower **Tolerance** to 5 pixels to flag both features. The overlay
draws the reference in amber, the sample in green, and violating
measurement locations in red.

**Contour profile tolerance** is under **Inspection → Measurements**.
Each input accepts a closed contour point list or a detected collection
with a selected item index. Use **Same frame** for contours measured
in one coordinate frame. Choose **Pixel grid** for contours from
separate images that are already aligned. When both frame sizes are
known, their pixel dimensions must match.

The block includes each vertex and subdivides longer contour segments
according to **Sample spacing**. It measures both directions. Each
profile is a list of `[x, y, signed_distance]` rows in source pixel
coordinates. Positive distances lie inside the opposite contour,
negative distances lie outside, and zero means a point lies on the
opposite outline. The maximum absolute deviation and RMS combine both
directions. RMS weights each sampled point equally, so vertex density
affects it. A point exactly on the tolerance limit passes.

Distances are sampled estimates using OpenCV's floating-point geometry.
The comparison supports up to 8,192 samples per contour and rejects
very dense pairs. Simplify dense contours or increase sample spacing
where spacing is the limiting factor.
[OpenCV shape analysis reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html)

```powershell
opencv-dataflow-run examples/applications/contour_profile/contour_profile.json
```

[All examples](../../README.md)
