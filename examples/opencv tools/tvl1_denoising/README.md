# TV-L1 denoising

Open [tvl1_denoising.json](tvl1_denoising.json) with **File → Open…**
and click **Run once** three times. The source replays aligned views
of a static target with different bright and dark impulse noise.
**Batch serial entries** collects the views. **TV-L1 denoise** fuses
them into one restored image and emits it with the middle view's
`observation_02.png` ticket.
In the interior of the bright patch, the mean absolute error falls
from 11.79 to 5.11 gray levels relative to the synthetic reference.

The block is under **Image processing → Filters**. Connect either
**Image** for a single view or **Observations** for an ordered batch.
All inputs must be aligned 8-bit monochrome images of the same size.
The output uses the middle observation's metadata, or the earlier
middle for an even batch. A mask or region limits where the restored
pixels are written, preserving the reference observation outside it.

**Lambda weight** balances fidelity to the observations against
total variation. **Iterations** sets the number of solver steps.
The block limits estimated native workspace and solver work so
large batches fail before allocation. Smaller batches, images, or
iteration counts reduce the requested work.
[OpenCV TV-L1 reference](https://docs.opencv.org/5.0/main_modules/photo_denoise.html)

```powershell
opencv-dataflow-run "examples/opencv tools/tvl1_denoising/tvl1_denoising.json" --all-images
```

[All examples](../../README.md)
