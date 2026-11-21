# Filter external stereo disparity

Open the [example project](external_disparity_wls.json) and run once. Both
filter blocks read the same noisy left disparity and aligned guide. The
paired block also reads a right-view disparity map and keeps estimates
with confidence of at least 0.2. The evaluation blocks compare each result
with a reference containing a sharp step from 8 to 16 disparity pixels.

The guide-only branch keeps all **13,168** source-valid samples, preserves
the depth edge and gives **0.0045 px** mean absolute error. With right
disparity, **11,669** samples pass the confidence check and give **0.0092 px**
mean absolute error. A band at the half-occlusion projects to valid but
inconsistent right samples. Right Support marks those locations as
available, while Confidence and Valid Mask reject them. An interior hole
in the right map removes support for the affected left samples.

Supply a rectified left guide on the disparity grid. Specify how many
integer samples represent one disparity pixel, and give any invalid code
in its original units. This project uses 32 samples per pixel and code
65535 on the left. The right TIFF uses signed float pixels and code âˆ’100.
Connect masks or a region to further restrict valid samples. Use the right
sign dropdown when a right-view matcher stores positive left-minus-right
values. Without a right map, the block emits filtered disparity and
validity. With a right map, it also emits right support and normalized
confidence. Missing output disparity is NaN.

The assets are original synthetic fixtures by OpenCVDataflowGUI
contributors under the project's MIT license. Run the example builder with
an empty output directory to regenerate them.
[OpenCV disparity WLS filter documentation](https://docs.opencv.org/5.0/extra_modules/classcv_1_1ximgproc_1_1DisparityWLSFilter.html)
