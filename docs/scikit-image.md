# scikit-image inspection tools

Install the optional tools with the application's Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[skimage]"
```

These blocks use scikit-image **0.26.x**, alongside OpenCV 5. They are under
**scikit-image → Measurements** and **scikit-image → Thresholding** in the
toolbox. Dependencies load when a block runs; a missing package produces a
contained block error. Existing skeleton and ridge tools remain available.

## Region properties

**Region properties (scikit-image)** (`skimage.region_properties`) wraps the
standard **44-property 2D regionprops catalog**, including grayscale and
per-channel color intensity measurements. Supply an integer label plane, or
choose Binary mode to label nonzero pixels with 4- or 8-neighbor connectivity.
An existing label ID is one region, even when its pixels are disconnected.
Sparse, negative and large int32 IDs are retained without allocating arrays
indexed by the largest ID. By default -1 and 0 are ignored.

The standard selection includes 19 commonly used properties. Comprehensive
adds all numeric, moment and tensor properties plus slices. Coordinates and
image patches are separate opt-ins because they can be large. Custom selects
an explicit list from this catalog:

| Family | Properties |
| --- | --- |
| Identity and geometry | `label`, `num_pixels`, `area`, `area_bbox`, `area_convex`, `area_filled`, `bbox`, `slice`, `centroid`, `centroid_local`, `coords`, `coords_scaled` |
| Shape and topology | `axis_major_length`, `axis_minor_length`, `eccentricity`, `equivalent_diameter_area`, `euler_number`, `extent`, `feret_diameter_max`, `orientation`, `perimeter`, `perimeter_crofton`, `solidity` |
| Moments and tensors | `inertia_tensor`, `inertia_tensor_eigvals`, `moments`, `moments_central`, `moments_hu`, `moments_normalized` |
| Intensity and weighted measurements | `intensity_min`, `intensity_max`, `intensity_mean`, `intensity_median`, `intensity_std`, `centroid_weighted`, `centroid_weighted_local`, `moments_weighted`, `moments_weighted_central`, `moments_weighted_hu`, `moments_weighted_normalized` |
| Cropped arrays | `image`, `image_convex`, `image_filled`, `image_intensity` |

An optional intensity image must match the label dimensions and, by default,
their known coordinate frame and ticket. Pixel-grid alignment explicitly
permits separately exported aligned images. Intensity units and channel order
are retained; alpha is excluded unless enabled. No implicit 8-bit conversion
occurs. No intensity image means intensity-dependent properties are unavailable.

`spacing_yx` and `offset_yx` are forwarded to native regionprops. Raw properties
retain native row/column conventions and property-specific scaling: area scales
by both spacings, centroids use spacing and offset, while bbox/slice stay local
pixel indices. The units label describes these measurements; it does not fit a
calibration. Some native properties, including perimeters and Hu moments, do
not support unequal row/column spacing. Undefined weighted centroids and the
low-order undefined entries of normalized-moment arrays can be nonfinite.

By default such values become JSON `null`, with a reason per property. Finite
components of an otherwise partially undefined array remain available. Error
mode instead fails the block when any requested property is unavailable.
Inspect `measurements.rows` or each collection item's `properties` and
`unavailable` fields for the full results.

Ordinary collection geometry, centroids, sizes, perimeter, angle and Hu metrics
remain in image pixels for compatibility with existing collection tools and
overlays. Native numeric properties also appear as flat metrics prefixed with
`regionprops.`. Choose **Metric → custom** in Filter/Sort object collection and
enter, for example:

- `regionprops.euler_number` to select rings;
- `regionprops.eccentricity` to rank elongation;
- `regionprops.intensity_mean.1` for the second intensity channel;
- `regionprops.moments_hu.0` for the first native Hu moment.

This custom-metric path also works with numeric metrics supplied by user blocks.
Retained collections can feed **Mask from filtered labels** using the block's
labels output and the same mask/region scope. Bounds and centroid graphics are
drawable objects and obey graphics/profile settings; `preview_count=0` disables
them. Requested numeric results remain available in Lean mode.

Region measurement is bounded to 2.1M label pixels, 10,000 regions, 16.8M
accumulated bounding-box pixels, and at most 2M requested property values.
The configurable defaults are lower. Budget violations fail before publishing
a partial collection. This block measures 2D images, not volumetric labels.

[Native regionprops reference](https://scikit-image.org/docs/0.26.x/api/skimage.measure.html#skimage.measure.regionprops)
· [Runnable region example](<../examples/skimage tools/skimage_region_properties/README.md>)

## Multi-Otsu

**Multi-Otsu threshold** (`skimage.multi_otsu`) estimates ordered class cutoffs
from the selected pixels. Outputs include int32 labels, thresholds, per-class
counts, the histogram, a color preview, and a binary mask combining the
`selected_classes` list. Class zero is a real class; -1 denotes pixels outside
the mask/region. Remove 0 from Region properties' ignored-label list to measure
the darkest class. For native-example compatibility, equality to a cutoff goes
into the upper class (`numpy.digitize`, `right=False`).

Choose 2–8 classes and 8–1024 bins. Integer levels are exact when their range
fits the bin count; otherwise the bounded histogram uses bin centers in raw
intensity units, including for uint16 images. This avoids native integer
histograms silently ignoring the bin limit. A 2M candidate-partition guard
rejects expensive combinations before native execution; reduce bins when using
more classes. Too few occupied bins is an explicit error.

## Advanced global thresholds

**Advanced global threshold** (`skimage.global_threshold`) provides **Otsu, Yen,
Li, Isodata, Minimum, Triangle and Mean**. A mask/region restricts both the
learning samples and output pixels. Bright polarity selects `image > threshold`;
Dark selects `image <= threshold`. Outputs are a uint8 mask, a scalar cutoff and
statistics, including the histogram, all Isodata candidates and Li's iterations.

Li exposes tolerance, a mean/median/quantile/explicit starting estimate and an
iteration limit. Minimum exposes its histogram-smoothing iteration limit.
Isodata lets you choose the lowest, middle or highest valid candidate. Histogram
methods expose bin count; Mean and Li operate on raw samples. Triangle uses
floating-point samples so its bin count applies to integer input too.
Convergence failures, unsuitable histograms and nonfinite results produce block
errors without silently switching algorithms.

## Advanced local thresholds

**Advanced local threshold** (`skimage.local_threshold`) provides **Gaussian,
Mean, Median, Niblack and Sauvola** with an odd `[height, width]` neighborhood.
Outputs are a uint8 mask, a float64 threshold map in raw units, and statistics.
Neighborhoods use the entire source image; masks/regions restrict only output.
Outside scope, both mask and threshold map are zero.

Gaussian/Mean/Median expose border handling and constant border intensity.
Gaussian exposes per-axis sigma or automatic window-derived sigma. Niblack
and Sauvola use native reflect boundaries and expose `k`; Sauvola additionally
exposes its standard-deviation reference range. Use a range appropriate to
the image units, for example 128 for uint8 or 0.5 for normalized [0,1] data.
scikit-image Niblack subtracts `k * std`; OpenCV's Niblack sign convention differs.
Offset subtracts from the computed threshold for every method.

All threshold blocks retain uint16 and floating-point intensity units.
RGB/BGR(A) inputs use weighted luminance, excluding alpha, without rescaling.
Extract a channel or convert explicitly before thresholding HSV/Lab. All support
up to 2.1M pixels, a 16384-pixel axis and finite intensities within ±1e12.
Median additionally bounds pixel-window work to 100M. Arbitrary generic local
functions and regionprops extra-property Python callables belong in user blocks.

[Native threshold reference](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html)
· [Runnable threshold example](<../examples/skimage tools/skimage_thresholding/README.md>)
