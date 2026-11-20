# Quasi-dense stereo correspondence

Open [the forward baseline](quasi_dense_forward.json) or [the reverse baseline](quasi_dense_reverse.json) and select **Run**. The pair contains two textured planes and an exposure change. Both graphs rectify the pair, exclude a patch from the right view, propagate matches and check the reconstructed depth.

**Quasi-dense stereo disparity** reports signed horizontal disparity as left X minus right X. The separate **Vertical Offset** output gives left Y minus right Y. Unmatched pixels use NaN and have a zero **Valid Mask**. The default vertical error limit is zero for calibrated depth. **Matches** exposes a bounded sample of accepted pairs for inspection.

The native matcher seeds correspondences with corners and pyramidal Lucas–Kanade tracking, then grows the matches using local correlation. The block checks each pair's square correlation footprint against the rectification ROIs and any masks or regions. The default reverse pass checks whether right-to-left propagation returns to the original left coordinate. **Support Mask** marks eligible left centers. **Valid Mask** marks centers with accepted pairs.

Each synthetic baseline produces **22,384 accepted matches** and covers **72.2%** of the reference-visible pixels. Three matched pixels differ from the reference by more than 0.5 px. The estimated disparities are ±6 and ±12 px, which reproject to planes at **3.0 m** and **1.5 m** with the corresponding calibration. In the saved reference rasters, zero marks pixels hidden in the other view or outside it.

The stereo images and references are original synthetic assets by OpenCVDataflowGUI contributors under the repository's MIT license. [OpenCV QuasiDenseStereo reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1stereo_1_1QuasiDenseStereo.html)

```powershell
python "examples/opencv tools/quasi_dense_stereo/build_example.py" --output D:\temp\quasi-dense
opencv-dataflow-run "examples/opencv tools/quasi_dense_stereo/quasi_dense_forward.json" --profile debug
```
