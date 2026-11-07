# Fourier gain mask

Open [fourier_gain.json](fourier_gain.json) with **File → Open…**, then
click **Run once**. The example loads a stripe image and a monochrome gain
image. The gain image has one black pixel at a stripe-frequency peak.
**Mask Fourier spectrum** mirrors the notch to its paired frequency.
**IDFT** reconstructs the broad brightness variation without the stripes.

**Mask Fourier spectrum** is under **Image processing → Spectral**.
Connect a DFT spectrum and a monochrome gain image the size of its
coefficient grid, including any padding. Unsigned gain images map
their full range to [0, 1]. Floating-point gains use their actual values,
so zero rejects a bin, one preserves it, and values above one amplify it.

Use **Centered** to match the magnitude preview or **Native** for zero
frequency at index zero. For real-image reconstruction, paired frequencies
need equal gains. **Require** rejects an asymmetric gain image. Choose
**Minimum** to mirror suppression, **Maximum** to mirror preservation,
or **Average** to blend the two gains. For a row-wise DFT, pairing occurs
within each row. The filtered spectrum retains the source image's
coordinate frame and ticket. [OpenCV array operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html)

```powershell
opencv-dataflow-run "examples/custom tools/fourier_gain/fourier_gain.json"
```

[All examples](../../README.md)
