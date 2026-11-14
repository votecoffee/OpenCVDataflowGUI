# Differential exposure comparison

Open a project with **File → Open**. **Compare exposure-adjusted images** is under
**OpenCV → Image Processing (imgproc) → Reference inspection**.

| Project | Expected result |
| --- | --- |
| [Local lighting](local_lighting.json) | Pass: remove smooth lighting differences on an already aligned synthetic texture. Compare about 38 DN raw RMSE with less than 1 DN adjusted RMSE. |
| [Local defect](local_defect.json) | Fail: retain all 20 injected defect pixels while correcting the background. Inspect contours, Defect mask and both residual maps. |
| [Global color](global_color.json) | Pass: recover known per-channel gains/offsets in a 16-bit BGR pair while preserving the image depth. |
| [Insufficient fit](insufficient_fit.json) | Fail: a small corner fitting region does not supply local models for the full inspection area. The ticket reports insufficient support. |

The aligned 192×128 images are original generated fixtures by OpenCVDataflowGUI
contributors, covered by the repository's MIT license. There are no third-party
or customer images. Source/scope comments are on each canvas; [truth.json](truth.json)
records formulas, coefficients and defect geometry.

**Image** is the unmodified candidate and the source of drawable result graphics.
**Adjusted** contains corrected samples at the original depth and color encoding.
Residual maps use floating raw sample units **before rounding/clipping** of
Adjusted, so integer saturation cannot conceal the measured discrepancy.
Consume **Valid mask** with the adjusted residual: unsupported pixels are zero
in that map and retain their original samples in Adjusted.

Without fitting inputs, inspection pixels also supply the fit. Use **Fit mask**
and/or **Fit region** to select independently known-good areas when defects could
affect the fit. Local models need nearby fitting support; Global can extrapolate
one model across the inspection region. Robust refits help sparse outliers but
cannot distinguish every broad defect from a lighting change. These examples
demonstrate software appearance compensation, not camera/radiometric calibration
or physical measurement qualification.

```powershell
python -m opencv_dataflow.runner "examples/opencv tools/differential_exposure/local_lighting.json"
python -m opencv_dataflow.runner "examples/opencv tools/differential_exposure/local_defect.json"
```

Failing inspections deliberately return a nonzero runner status. Regenerate all
five PNGs, the truth record and four graphs without network access:

```powershell
python "examples/opencv tools/differential_exposure/build_example.py"
```

See [the fitting and inspection contract](../../../docs/differential-exposure.md).
