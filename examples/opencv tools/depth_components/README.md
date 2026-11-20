# Depth component measurement

Open [depth_components.json](depth_components.json) and run once.
The depth raster describes three parts above a background. Two parts
touch in the image. A 35 mm local depth-step limit keeps them separate,
and an image-pixel cutoff removes a two-pixel speck.

| Part | Depth | Visible width × height | Source pixels |
| --- | ---: | ---: | ---: |
| 1 | 1,000 mm | 160 × 112 mm | 876 |
| 2 | 1,250 mm | 200 × 130 mm | 832 |
| 3 | 900 mm | 81 × 50.4 mm | 252 |

The width filter selects Part 2 at a minimum of 190 mm.
**Mask from filtered labels** uses the resulting IDs and the exact
Labels output to rebuild its 832-pixel mask. The hole in Part 1 remains
empty. In Debug, the block renders a color label preview and draws the
visible dimensions. Both profiles retain the collection, masks and
measurements.

The dimensions describe the visible footprints along camera axes.
Projected Area is the sum of the measured pixels' fronto-parallel
footprints. Use calibrated undistorted pinhole depth and supply a
foreground mask or region when the scene requires additional selection.
Set Maximum Depth Step to zero to see the two touching parts merge
under ordinary spatial connectivity.

The synthetic depth raster is by the OpenCVDataflowGUI contributors
(MIT). Rebuild in a fresh directory with
`python build_example.py --output <directory>`.
[SciPy sparse graph connectivity reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html)
