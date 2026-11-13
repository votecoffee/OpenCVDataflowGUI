# Comprehensive region measurements

Install `.[skimage]`, then open `skimage_region_properties.json` and press
**Run once**. No plugin or external images are needed.

The original 16-bit RGB fixture stores three label IDs in R and varying
intensities in G/B. Extract R, then inspect **All numeric region properties →
Results** for the 38 numeric/moment/tensor/slice properties. Enable coordinates
and image patches to request all 44 standard 2D properties. These can be large.

Native measurements use 0.1 mm pixel spacing. Collection geometry and graphics
remain in image pixels. Compare counts, pixel centroids and topology against
`expected.json`: the rectangle has 1200 pixels; the ring alone has Euler number
zero. The filter retains that ring, reconstructs its exact mask and selects an
annotated original image for the filmstrip. The parallel sort ranks regions by
physical area through the custom metric `regionprops.area`.

The two image inputs share one acquisition, so their frame and ticket agree.
Change **Unavailable → error** to reject undefined requested properties rather
than reporting nulls and reasons. Normalized moments have some structurally
undefined entries even for valid objects.

Fixture and graph are original OpenCVDataflowGUI work under the repository MIT
license. Algorithm reference: [scikit-image regionprops](https://scikit-image.org/docs/0.26.x/api/skimage.measure.html#skimage.measure.regionprops).
See [tool contracts and catalog](../../../docs/scikit-image.md).
