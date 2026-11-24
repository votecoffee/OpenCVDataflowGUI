# Static saliency maps

Open [static_saliency.json](static_saliency.json) and choose **Run Once**. Both
branches look for prominent regions in a synthetic inspection scene with one
high-contrast part and lower-contrast clutter. The output image shows mask
outlines and the strongest response. The two **Overlap** blocks compare the
thresholded masks against the known part silhouette.

| Method | Pixel-ranking AUC | Mask intersection over union |
| --- | ---: | ---: |
| Spectral residual | 0.986 | 0.543 |
| Fine-grained | 0.993 | 0.597 |

The ranking score measures how often a part pixel scores higher than a
background pixel, with half credit for ties. Both masks use the 94th-percentile
cutoff. Ties at that value can affect how many pixels are selected.

**Static saliency map** appears under **OpenCV → Saliency → Static saliency**.
Both methods operate on uint8 luminance and emit a float32 [0,1] response and
a reusable 0/255 mask on the original image grid. Set the spectral working
width and height to tune the frequency analysis. Fine-grained saliency uses
local intensity contrast at its native scales. Use **Otsu**, **Fixed** or
**Percentile** to turn scores into a mask. Otsu uses rounded 8-bit scores.

The block analyzes the full frame. Masks and regions limit the reported
scores, peaks, and threshold statistics to the selected pixels. Uniform
luminance and empty selections produce blank maps and masks.
[OpenCV spectral residual](https://docs.opencv.org/5.0/extra_modules/classcv_1_1saliency_1_1StaticSaliencySpectralResidual.html)
[OpenCV fine-grained saliency](https://docs.opencv.org/5.0/extra_modules/classcv_1_1saliency_1_1StaticSaliencyFineGrained.html)

The scene and silhouette are original MIT assets. Rebuild from the repository root.

    python "examples/opencv tools/static_saliency/build_example.py" --output "examples/opencv tools/static_saliency"
