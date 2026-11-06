# Hann window

Open [hann_window.json](hann_window.json) with **File → Open…** and
click **Run once**. Each row contains a sinusoid with 6.35 cycles
across 128 pixels, so the signal falls between Fourier bins. Compare
the two **Magnitude** previews. The Hann taper reduces energy far
from the main peaks and broadens the peaks near the signal.
In this example, the share of spectral power beyond ±12 bins
falls from 0.281% to 0.00036%.

**Hann window** is under **Image processing → Spectral**. Use
**Rows** to taper each row independently or **2d** to taper both
dimensions. The **Window** output contains the coefficients. The
**Image** output contains the windowed samples, and **Coherent
gain** reports the mean coefficient. Absolute spectral amplitudes
depend on the taper.

The default **Hann** mode squares OpenCV's square-root window to
produce the standard separable Hann coefficients. **Sqrt hann**
matches the coefficients used by OpenCV phase correlation. Use
matching float precision when wiring the window to a phase
correlation block. Unsigned images are normalized to [0, 1] before
the taper, matching the DFT block's input scaling.
[OpenCV phase correlation and window reference](https://docs.opencv.org/5.0/main_modules/imgproc_motion.html)

```powershell
opencv-dataflow-run examples/image_processing/hann_window/hann_window.json
```

[All examples](../../README.md)
