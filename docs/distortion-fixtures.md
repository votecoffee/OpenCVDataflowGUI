# Controlled image distortion fixtures

**Custom > Image processing > Test fixtures > Generate image distortion fixture**
warps a supplied image with a declared analytic sampling model. It supports
uint8, uint16, int16, float32 and float64 grayscale or color images, preserving
depth and encoding. All channels, including alpha, are sampled independently;
the tool does not perform alpha-aware compositing. Output size equals input size.
The limit is 4,194,304 pixels with both axes below 32767.

Use the [original part example](<../examples/custom tools/distortion_fixture/README.md>)
to test four independently checked shear, perspective, lens-like and seeded
branches. A diagnostic fixture is synthetic ground truth, not a fitted camera
calibration or an estimate of a real part's registration.

## Mapping direction and model

Every output pixel `(x,y)` selects a donor in the input image. **Map X/Y** are
float32 donor coordinate planes on the output grid. **Mapping** records the
effective coefficients, normalization, source/output frames, sampled minimum
Jacobian, float32 coordinate rounding and seed. **Point pairs** provides
continuous float64 output/input coordinate pairs, independent of map rounding.
The optional points input overrides the parameter; a wired empty list disables
probes. Without either, nine output-grid probes are generated. Result graphics
mark these output positions, using the output coordinate frame.

Let `cx=center_x*(width-1)`, `cy=center_y*(height-1)` and
`R=radius_scale*max(width,height)`. Center parameters are fractions. Normalize
`u=(x-cx)/R`, `v=(y-cy)/R`, then evaluate:

```text
d = 1 + perspective_x*u + perspective_y*v
a = (u + shear_x*v) / d
b = (v + shear_y*u) / d
r2 = a*a + b*b
m = 1 + k1*r2 + k2*r2*r2
input_x = cx + R*(a*m + 2*p1*a*b + p2*(r2 + 2*a*a))
input_y = cy + R*(b*m + p1*(r2 + 2*b*b) + 2*p2*a*b)
```

The order is projective/shear, radial/tangential, then pixel denormalization.
These are **output-to-input** equations. A positive shear coefficient changes
the donor direction; it is not a forward displacement of source features.
No forward transform or inverse is guessed. Denominators at or below 0.05,
nonpositive/near-singular sampled Jacobians and donor magnitudes above 32760
are rejected. Positive sampled Jacobians do not establish global invertibility.

Fixed mode uses signed configured coefficients. Uniform-bounds mode requires
nonnegative bounds and draws one value from `[-bound,+bound]` for each of
`shear_x, shear_y, perspective_x, perspective_y, k1, k2, p1, p2`, in that order.
Each activation creates a private NumPy PCG64 generator. A wired seed overrides
the parameter. Equal seed/configuration repeats across serial or parallel
branches and does not advance global random state. Unsafe random draws fail;
the tool does not silently redraw and change the requested distribution.

## Support, interpolation and native limits

The entire image is warped. Optional source masks/regions constrain **Valid
mask**, rather than blending warped and unwarped coordinate systems. The mask
requires in-bounds donors and source support eroded by a conservative radius:
zero for nearest, one for linear, two for cubic and four for Lanczos4. This
nominal halo can exclude border pixels even for an identity warp. Border fill,
reflection, wrapping and replication never establish original source support.
**Valid fraction** reports retained support; generating a low-coverage fixture
is not itself an inspection failure.
Validity belongs to the fixture's selected kernel. Changing a downstream
remapper's interpolation requires a matching support mask.

The pinned OpenCV 5 native float64 cubic implementation truncates fractional
boundary samples, including an identity mapping of a constant 1.5-valued image.
The fixture and ordinary floating/packed remappers reject this combination.
Choose linear interpolation or explicitly convert to float32. Other supported
depth/interpolation combinations are checked against native resampling and
independent references.

Packed maps round donor coordinates to OpenCV's fixed-point grid. Their pixel
values can differ from floating maps, especially at sharp edges. The native
qualification fixture measured up to 179 counts of difference on a uint16 ramp
with a constant-border discontinuity. That value is an observation, not a
general bound. Tests instead use the coordinate-rounding bound and the fixture's
sample range to bound bilinear differences, and compare packed execution with
its own native packed-map reference.

## Coordinate ownership and map reuse

The warped image and map planes share a new coordinate frame. The old spatial
and pyramid context is recorded in the model; it is not left attached as a
false affine relationship. Tickets, acquisition start times and arbitrary
companion payloads stay on current connections.

Wire **Map X/Y** to ordinary **Remap image coordinates**, along with an image
on the original source grid, to reproduce the warp. **Pack remap coordinates**
and **Remap with packed coordinates** also preserve its output frame. Typed
maps reject swapped axes, mismatched pairs, changed output grids and different
source frames. The image being remapped owns its current ticket and timing;
the maps define geometry, not a new acquisition identity.

Plugins can provide the same optional `coordinate_remap` payload contract:
`model=image_coordinate_map_v1`, a common `id`, `axis=x/y/packed`, source/output
sizes and source/output spatial records. Both planes must carry matching
provenance and their original output spatial grid. A packer retags the axis
as `packed`. The producer is responsible for honest provenance when changing
map values. Plain exported image planes lose packet metadata; this contract
does not make arbitrary map images self-describing.

The existing lens-specific map protocol and untagged numeric maps remain
supported. Saved projects store controls and layout and regenerate maps.

Native reference: [OpenCV geometric transformations](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html).
