# Stereo WLS confidence

Open [stereo_confidence.json](stereo_confidence.json) with **File → Open…**
and click **Run once**. The graph masks a vertical strip in the right view
of a rectified synthetic stereo pair. Two disparity blocks apply weighted
least squares (WLS) filtering. One uses a confidence cutoff of 0.8 and the
other leaves the cutoff at zero. The cutoff reduces valid estimates from
51,840 to 44,819 on the bundled pair. The gated disparity reconstructs the
plane at a median depth of 3 m.

Turn on **Wls** in **Stereo BM / SGBM disparity** to emit a float32
**Confidence** image in the left view's pixel grid, normalized to [0, 1].
The image reports the native WLS confidence independently of the selected
cutoff. **Minimum Wls Confidence** invalidates lower-confidence estimates
in both **Disparity** and **Valid mask**. Invalid disparity values are NaN.
A cutoff of zero retains the original WLS validity behavior.
[OpenCV WLS confidence reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1ximgproc_1_1DisparityWLSFilter.html)

**Wls Lr Threshold** controls the left-right consistency tolerance in
sixteenths of a pixel. The default of 24 corresponds to 1.5 pixels.
**Wls Depth Radius** controls the low-confidence neighborhood around
discontinuities. The default of -1 uses OpenCV's matcher-specific radius.
The confidence port emits only when WLS is enabled.

```powershell
opencv-dataflow-run "examples/opencv tools/stereo_confidence/stereo_confidence.json"
```

[All examples](../../README.md)
