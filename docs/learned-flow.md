# Learned optical flow

`ml.optical_flow` runs paired-image ONNX models and emits a typed dense field
on the original **first-image grid**. Each vector is `(dx,dy)` from first to
second: a first pixel `(x,y)` corresponds to second `(x+dx,y+dy)`. It is displacement
in original pixels, not velocity. The initial trained baseline uses unchanged
RAFT Sintel weights through OpenCV 5 Classic/CPU. See the
[twelve projects](<../examples/opencv tools/learned_flow/README.md>).

## Inputs and preparation

Supply equal-size first/second images with corresponding pixel grids. Pixel grid
mode accepts independently identified frames; Same frame rejects known spatial
conflicts using the common motion-tool check. Neither mode estimates registration,
verifies camera calibration nor proves temporal exposure order. Inputs may have
different tickets, as consecutive video frames usually do. Output image, field
and measurements retain the first image's ticket, spatial/pyramid context and
arbitrary payload. `flow_pair` also retains both frames' ticket, grid, pyramid and
recipe references. The first image's original pixels/depth/color/alpha are returned
unchanged. To estimate reverse flow, swap inputs; negating a field does not move
its index grid and is generally not a reverse field.

Accepted ordinary images are 2D gray or RGB/BGR/alpha, uint8/uint16 or finite
float32 [0,1]. Integer full scale becomes unit float32. Preparation converts color,
applies identical stretch or rounded letterbox, then `(unit_pixel - mean) / std`.
Mean/std are per-frame channel lists. Inputs are either distinct named tensors or
one concatenation with first channels followed by second channels, explicitly
packed as NCHW or NHWC. Full images remain context even outside selected writes.

RAFT's published model accepts raw RGB float values in [0,255]. Its prepared
contract therefore uses zero means and std=1/255 on unit RGB. Selected full-size
output is `12006`. The model also exposes a smaller `12007` head; native terminal
ordering differs between engines, so output selection always uses the name.
Native loading preserves the requested engine through the common snapshot loader.
No training, conversion, remote Python execution or implicit download occurs.

## Vector and support contracts

Declare a batch-one, two-channel head in NCHW, NHWC, CHW or HWC layout. Component
order is xy or yx. Once decoded to xy, apply fixed per-component scale and offset:
`q_xy = decoded_xy * head_scale + head_offset`. Both scale entries must be nonzero.
Units are explicit:

| Head units | Source-pixel multipliers for dx,dy |
| --- | --- |
| network_pixels | source_width/content_width, source_height/content_height |
| head_pixels | network_width/head_width * source_width/content_width, network_height/head_height * source_height/content_height |
| source_pixels | 1, 1 |

Rounded letterbox content uses its actual width/height independently. A single
average scale would give incorrect vectors under anisotropic resize. Source
pixel centers map to head samples with the half-pixel rule:

```
u = ((x + 0.5) * content_width/source_width + pad_left) * head_width/network_width - 0.5
v = ((y + 0.5) * content_height/source_height + pad_top) * head_height/network_height - 0.5
```

Head cells touching padding are unsupported. An enabled raw sentinel in either
component invalidates the vector. Affine values exceeding per-axis Maximum motion
are invalidated before interpolation; bounds are checked again afterwards.
Linear resampling requires every nonzero-weight contributor, while Nearest checks
its chosen cell. Invalid vectors contain NaN in both components and have a zero
entry in the uint8 Valid mask. Legitimate zero motion remains valid unless an
explicit sentinel setting excludes it. Values are never clipped into acceptance.

First mask and region intersect as a write selection on the first grid. Second
mask is checked at first+flow on the second grid. Every nonzero bilinear mask
contributor is required, including tiny fractional weights. Known selection-frame
conflicts reject execution. Endpoint bounds are checked by default and are always
enforced when a Second mask is supplied. These checks establish geometric support,
not visibility, occlusion freedom or learned confidence. Minimum count/fraction
controls ordinary coverage pass/fail; the score is retained selected coverage.
Empty selection returns current empty coverage without model I/O.

## Existing consumers and persistence

Flow is a `DenseFlow` compatible object with explicit first-to-second description.
By default, `flow_association` uses the existing content-bound flow manifest:
source-to-target direction, exact vector hash and both original image hashes,
encodings, dimensions and spatial/pyramid records. **Warp image with dense flow**
can consume it as Source to target, using first as Image, second as Target and the
Valid mask as Flow support. The warp estimates an inverse sampling map; it does
not establish true inverse/occlusion accuracy. A wrong direction or changed image
anchor rejects reuse. Disabling Bind image content explicitly removes these
consumer association checks; the pair description and frame references remain.

**Save dense flow file** / **Load dense flow file** retain float32 values and
direction/image bindings through native `.flo` transport. Select Preserve invalid
vectors when saving a masked field. The arithmetic project verifies exact
supported warp pixels and a test verifies NaN-preserving native round trips.
Other consumers can have stricter contracts: the current variational refinement
tool requires a finite initial field and cannot directly consume masked NaNs.
This increment does not change that tool's contract.

The shared sealed ONNX contract binds weights, parameters and preparation.
Save As includes exact model bytes and license notices; JSON/XML reopen preserves
frame order and settings. Source pixels, combined prepared input elements, head
axes and total output elements have explicit guards. Native/shape errors clear
the cache; the base executor records the exception without a new output event.

## Results, overhead and evidence

Preview is a diagnostic HSV direction/magnitude map. Drawable arrows reference
the first image and can be combined with other result graphics. Requested spacing
is increased if needed to cap 2,048 arrows. Display magnitude never changes
numerical flow. Magnitude and angle are available as optional ports; angle uses
degrees in [0,360). Lean/graphics-off omits preview/drawings and skips unexposed
numeric display maps. Explicitly exposed Magnitude/Angle still compute in Lean.
Observation settings never change the field or support mask.

Nine arithmetic projects exercise paired ramp motion, subpixels, packed inputs,
low-resolution heads, anisotropic scaling, rounded letterbox, selections, endpoint
masks and deliberate coverage failure. Three native projects use translated copies
of a CC BY 4.0 T-LESS image. Their controlled shifts are (4,-2), (-3,2) and (0,0),
not recorded optical motion or physical part motion. On a fixed 200x140 interior,
independent named-head inference and four-contributor source mapping report mean
endpoint errors **0.01861**, **0.01457** and **0.00297 source pixels**. All three
retain full evaluation support with no endpoint errors over one pixel there.
These results do not establish accuracy on other scenes, occlusions, changing
illumination or accelerators. [Sources and exact hashes](<../examples/opencv tools/learned_flow/SOURCES.md>)
and the saved independent benchmark document the qualified scope.
