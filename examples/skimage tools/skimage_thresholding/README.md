# Multi-Otsu and advanced thresholding

Install `.[skimage]`, open `skimage_thresholding.json`, and press **Run once**.
The original uint16 fixture has three material bands with known base intensities,
a vertical illumination ramp and seeded noise. `expected.json` records the bands.

**Three material classes** exposes raw cutoffs, class counts, a combined mask
for class 2, and a color preview selected for the filmstrip. The connected region
block measures all classes, including zero. Cutoffs use a bounded 128-bin
histogram; exact equality belongs to the upper class. This is intensity
classification, not connected-component labeling.

Compare Yen and Li global masks with Gaussian, Niblack and Sauvola local masks.
Local tools expose their threshold maps as well as binary results. Sauvola's
range is 32768 for this 16-bit demonstration; it is a raw-unit tuning parameter.
No input is reduced to 8 bits. Methods, border modes and polarity use dropdowns.
Wire a mask or region to limit the selected output. Global estimators also learn
only from that selection; local neighborhoods still use the full image.

Fixture and graph are original OpenCVDataflowGUI work under the repository MIT
license. Algorithm reference: [scikit-image thresholds](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html).
See [parameters and behavior](../../../docs/scikit-image.md).
