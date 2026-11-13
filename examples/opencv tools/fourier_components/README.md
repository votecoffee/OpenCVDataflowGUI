# Fourier magnitude and phase reconstruction

Open `project.json` to compare full, phase-only and magnitude-only reconstruction
of an original 16-bit color bracket illustration. Run once and inspect each
block's image and results. The contrast-stretched preview helps show signed,
low-amplitude results; it does not change the raw output.

The projects require only the built-in tools:

| Project | Purpose |
| --- | --- |
| `project.json` | Reconstruct each RGB channel of the bracket using three component modes. |
| `verification.json` | Compare three color-impulse reconstructions with independently assigned reference pixels; every maximum absolute error must be below `1e-12`. |
| `spectrum_pipeline.json` | Use the existing DFT and IDFT blocks around the new typed Fourier component block, with optimal padding and inspectable magnitude, phase and phase-validity outputs. |

## Controls and output values

**Reconstruct Fourier image components** (`cv.reconstruct_fourier_image`)
accepts grayscale or RGB(A)/BGR(A) images. Choose independent color channels or
grayscale, a full 2D or per-row transform, exact or optimally padded dimensions,
DC retention/removal, and floating precision. Padding applies only to transformed
axes; the inverse uses the padded length for scaling and crops to the original size.

Unsigned samples are divided by their type's full scale (`255` or `65535`).
Floating input keeps its values. The default output is signed floating point.
Optional unit-RMS or min/max normalization can use independent channel scales
or one shared color scale. A flat min/max result becomes zero. Color output
retains RGB/BGR order and omits alpha; **Source** preserves the original image,
including alpha, depth and payload.

- **Full** retains the selected source samples and both spectral components.
- **Phase only** divides reliable complex bins by their magnitude. Bins must
  strictly exceed both the absolute and relative floors; other bins become zero.
  The relative floor uses the peak after DC removal, per row in row mode.
- **Magnitude only** retains magnitude and sets phase to zero. It creates a new
  origin-based coordinate frame: source positions no longer describe this image.
  The original spatial and pyramid metadata remain available as source context.

Phase/full images keep source spatial and pyramid provenance. Tickets and other
companion payload data are retained in all three modes.

Optional mask and region inputs intersect to select **source samples before the
global transform**. Samples outside the selection become zero. The reconstructed
image may have values outside that selection; this is not an output clipping mask
or a sliding local transform.

**Fourier magnitude and phase components** (`cv.fourier_components`) accepts a
typed monochrome DFT spectrum. Its modified spectrum remains in native bin order
and requests floating IDFT output, preserving negative values. The diagnostic
images show the DC-adjusted incoming spectrum: magnitude, phase in radians
(`[-pi, pi]`), and a separate phase-validity mask. Undefined phase is displayed as
zero. Choose centered or native diagnostic layout; row mode shifts only x.
Diagnostic images have a frequency grid separate from source image coordinates,
with x in cycles per current pixel and y in either frequency or source rows.

Real reconstruction requires conjugate symmetry. **Require** accepts native
precision roundoff and cleans paired bins; **Average** explicitly repairs an
arbitrary spectrum. Phase thresholds do not discard coefficients in full or
magnitude-only mode. Both tools limit the padded transform to 2.1 million pixels
and 16384 per axis; larger images need an explicitly selected or resized input.

## Independent reference and attribution

`assets/truth.json` records the RGB impulse positions and amplitudes. Full output
must reproduce the input divided by `65535`. Phase-only output must contain unit
impulses at the original positions. Magnitude-only output must place the original
unit-range amplitudes at `[0, 0]`. These reference images use direct pixel
assignments, with no FFT implementation. `verification.json` uses ordinary image
arithmetic to convert their stored counts to floating units before comparison.

All images, reference data, projects and `generate.py` are original work covered
by this repository's MIT license. Regenerate with:

```powershell
.\.venv\Scripts\python.exe "examples/opencv tools/fourier_components/generate.py"
```

Algorithm reference: [OpenCV DFT/IDFT documentation](https://docs.opencv.org/5.0/main_modules/core_array.html).
