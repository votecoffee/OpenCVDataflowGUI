# Motion deblur

Open [motion_deblur.json](motion_deblur.json) with **File → Open…**
and click **Run once**. **Wiener deblur** restores a frame that was
blurred by a known eight-pixel horizontal motion path. The two
**Image quality** blocks compare the blurred and restored frames
against the same sharp reference. RMSE falls from 17.45 to 6.22 gray levels.

In **Motion** mode, set **Motion length** in pixels and **Motion
angle** in degrees clockwise from horizontal. The model assumes
constant motion along a centered line. The example uses circular
boundaries to match the generated input.

In **Defocus** mode, set **Defocus radius** in pixels to approximate
a circular out-of-focus blur. Both models produce normalized kernels
on the **Psf** output. The effect depends on how closely the selected
kernel matches the image degradation.
[OpenCV motion deblurring tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/motion_deblur_filter/motion_deblur_filter.html)

```powershell
opencv-dataflow-run examples/motion_deblur/motion_deblur.json
```

[All examples](../README.md)
