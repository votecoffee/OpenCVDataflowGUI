# Nonlinear image and selection remapping

**Remap image and selection** (`cv.remap_selection`) applies the same
destination-to-source X/Y maps to an image and its entire selection. The packed
variant (`cv.remap_selection_packed`) accepts **Pack remap coordinates** output.
Use these blocks after **Build lens correction maps** to carry a source ROI,
including holes and disconnected components, into the corrected image. They also
accept untagged generic maps. Existing ordinary Remap blocks retain their ports
and defaults.

Connect the original image and optional source `mask` or `region`. If both are
connected they intersect. Mask values are uint8, one channel, with any nonzero
value selected. Shapes, composite expressions, transformed regions and exact
raster regions are supported. Known source coordinate frames and dimensions must
match; map components from different lens corrections are rejected. Declared
map metadata is trusted, not an authentication of externally modified arrays.

## Sampling and outputs

The image output is the ordinary native remap result for the chosen
**Interpolation**, **Border** and scalar **Border value**. Optional **Border values**
overrides the scalar with a JSON list in stored channel order, such as RGB for an
RGB image. An empty list uses the scalar. Integer depth
conversion and saturation follow native image sampling. The image is not masked
or painted with annotations.

| Output | Meaning |
| --- | --- |
| Image | Remapped image, retaining its depth and color encoding. |
| Mask | Binary uint8 target-grid inspection selection, zero or 255. |
| Region | Exact JSON selection on the same target grid; optional emission. |
| Coverage | Float32 nearest/linear source-selection contribution, from zero to one. |
| Valid mask | Full real-source support for the image sampling footprint. |
| Report | Counts, grids, policy, lens identity and estimated working allocation. |
| Result graphics | Optional drawable selected boundaries, in corrected coordinates. |

**Selection rule** has two choices:

- **Complete footprint**, the default, requires the image sampling footprint to
  lie entirely inside the selection and real source image.
- **Coverage threshold** accepts the configured nearest/linear contribution above
  **Minimum coverage**, still requiring the complete image footprint to have real
  source support. A zero threshold accepts any positive contribution.

Nearest image interpolation checks nearest-pixel support. Linear checks full
bilinear support. Cubic and Lanczos4 use conservative selection/image erosion
halos of two and four source pixels, followed by full bilinear support. These
halos can exclude extra pixels; they avoid interpreting negative interpolation
weights as fractional selection coverage. Coverage always uses the separately
chosen nearest/linear policy, regardless of image interpolation.

Replicated, reflected or wrapped image borders never add inspection validity.
Subsequent selection inversion or replacement is deliberate new geometry: retain
and intersect **Valid mask** if those downstream selections must exclude filled
borders too.

On the qualified OpenCV 5 build, floating X/Y maps retain finer linear fractions
than packed maps. Packed subpixel maps use 1/32 phases, so image samples, coverage
and selected boundaries can differ slightly. Packed nearest maps discard
fractional coordinates completely; original subpixel coverage cannot be recovered.
The examples check each representation against its own independent reference.

## Coordinate ownership and measurements

All outputs carry the source ticket and metadata with a new target spatial grid.
Lens/coordinate map producers supply that grid when known. Untagged maps create
an independent grid identified by source spatial context, target size and map
content. A nonlinear correction cannot be represented by the source's single
affine `to_root` matrix; source pyramid and map-output tags are removed. The
report retains source spatial context and lens identity for traceability.

Connect output Image and Region to **Measure selected region**, or Image and
Mask to **Image statistics**. Measurements count target pixels. A pinhole
profile's units do not turn those counts into physical area; calibrated planar
or surface-area integration is a separate operation. Ordinary uint16/float scalar
images can be remapped; physical `DepthArray` geometry is rejected by the common
image contract and is not reconstructed here.

**Minimum selected pixels** defaults to one. An empty supported selection gives
an ordinary failed inspection with outputs and a reason retained; zero permits
an empty selection. Invalid maps/frames/settings are contained block exceptions
with no current output event. There is no training or remembered processing state.

## Exact regions and resource limits

Region output uses `image_region_v1`, `shape: "raster"`,
`encoding: "scanline_runs_v1"`, `image_size: [width,height]`, boolean `invert`, and
sorted nonoverlapping `[row,start,length]` integer runs. This preserves holes
exactly rather than approximating them with contour polygons. Existing selection
consumers, region composition and affine/projective **Transform region** accept
it. Rasterization shares a maximum of one million runs across nested children,
alongside the existing 16-level/128-node expression guards. Runs must fit the grid.

**Emit region** can be disabled to skip encoding on noisy masks. Its default
100,000-run output limit is separate from the hard shared rasterization limit.
Source/output axes are capped at 16,384 pixels and each grid defaults to a
16-million-pixel budget. **Maximum working bytes** checks an array estimate before
map unpacking and image sampling; it is not a process RSS or native allocator
guarantee. Runtime input copies, expression rasterization, Python object overhead
and native workspaces can require additional memory.

Result-graphic contours use at most 262,144 preview pixels and a configurable
point budget; their decimation never changes numerical Mask, Region or Coverage.
Lean mode skips graphics and permits readonly input sharing. The inspector exposes
dropdowns, validation tooltips, preview/run and XML persistence through the common
block contracts.

See [five original projects](<../examples/opencv tools/selection_remap/README.md>),
[regions](regions.md), [image formats](image-formats.md) and
[native OpenCV remap](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).
Pinhole distortion is described in the
[OpenCV calibration module](https://docs.opencv.org/5.0/main_modules/calib.html).
