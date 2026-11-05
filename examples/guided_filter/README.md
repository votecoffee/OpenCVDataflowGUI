# Guided edge-preserving filter

Open [guided_filter.json](guided_filter.json) with **File → Open…**, then click
**Run once**. **Guided filter** uses the bundled noisy grayscale image as its
own guide. In the uniform left region, the standard deviation drops from about
8.9 to 1.3 gray levels. The step across the center remains about 111 gray
levels.

An optional guide can come from a different processing branch in the same pixel
frame. **Radius** controls the neighborhood, **Epsilon** controls the amount of
regularization, and **Scale** sets the internal subsampling fraction. Integer
images are normalized to [0, 1] before filtering. **Output depth** defaults to
the source depth. The block preserves alpha and supports limiting writes to a
connected mask or region. [OpenCV guided filter reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_filters.html#guidedfilter)

```powershell
opencv-dataflow-run examples/guided_filter/guided_filter.json
```

[All examples](../README.md)
