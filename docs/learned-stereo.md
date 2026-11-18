# Learned stereo disparity

`ml.stereo_disparity` runs a local paired-image ONNX model and emits disparity
on the **original left-image grid**, in original-image pixels. Its output
convention is always `d = x_left - x_right`. The initial trained baseline is
unchanged HITNET ETH3D ONNX on OpenCV 5 New/CPU. The
[twelve projects](<../examples/opencv tools/learned_stereo/README.md>) include
native known-shift controls and an exact calibrated reprojection workflow.

## Pair preparation and identity

Supply equal-size, horizontally rectified left/right images. The block does not
estimate rectification or verify epipolar alignment. Use the existing **Rectify
stereo images** tool with a calibrated rig when needed. An optional
`stereo_rectification_v1` input checks original size, finite invertible Q, units
and frame, then passes that record through for **Disparity to 3D cloud**. This
checks a declared record, not the correctness of its calibration or image pairing.

Accepted sources are uint8/uint16 or finite float32 [0,1], with ordinary gray,
RGB/BGR or alpha encodings. Integer full scale becomes unit float32. Each view
is converted to the selected model color order, resized identically, padded if
letterboxed, and normalized as `(unit_pixel - mean) / std`. Mean/std are per-view
channel lists. Alpha is excluded from inference. `image` returns the original
left pixels/encoding/depth/alpha unchanged.

Input mode selects two distinct named bindings or a single binding containing
left channels followed by right channels. NCHW/NHWC packing is explicit. The
selected output must be a left-reference, batch-one, channel-one disparity head
in NHW, NCHW, NHWC, HW, CHW or HWC layout. Other head meanings, such as a right-grid
map or vector flow, require a different decoder and must not be relabeled here.

The default ticket policy requires both views to have the same session, kind and
value whenever either is ticketed. Same-value mode explicitly compares kind/value
across sessions; Unchecked disables pairing validation. Both unticketed is allowed
and reported unverified. Matching tickets do not establish simultaneous exposure.
The left input owns the output frame, ticket, arbitrary payload and pyramid
context. `stereo_pair` retains both views' ticket, spatial, pyramid and recipe
references without forcing their different camera grids to share an origin.

## Quantity and geometry

First transform the native head by the fixed affine rule
`q = head_scale * head + head_offset`. Select its units:

| Head units | Multiplier to source-pixel disparity |
| --- | --- |
| network_pixels | source_width / resized_content_width |
| head_pixels | network_width / head_width * source_width / resized_content_width |
| source_pixels | 1 |

Right-minus-left heads additionally change sign. These choices are model
declarations, not inferred from tensor dimensions. For a sixteenth-pixel encoded
head, use a fixed scale of 1/16 in its actual grid units. No display normalization
or automatic metric fitting changes the numerical result.

Source center x maps into head coordinates as:

```
u = ((x + 0.5) * resized_content_width / source_width + pad_left)
    * head_width / network_width - 0.5
```

The vertical rule is identical. Network and head sizes may differ, including
rounded letterboxes. Head cells must lie wholly within unpadded content. Optional
raw sentinels and source-pixel disparity bounds invalidate contributors before
interpolation. Linear resampling requires every nonzero-weight contributor;
nearest checks the selected cell. Bounds are checked again after resampling.
Invalid output is NaN, with a matching uint8 `valid_mask`; values are not clipped.

Left mask/region define the output write selection. Both full images remain model
context. The right mask is checked at the predicted `x_right = x_left - d`, on
the right camera's own grid. Fractional positions require both neighboring mask
samples. Known mask/region frame conflicts are rejected. Correspondences outside
the source are invalid by default; a supplied right mask always enforces that
boundary even if the optional boundary check is disabled. This support test does
not establish photometric agreement, occlusion freedom or matching accuracy.

Minimum valid count/fraction controls an ordinary coverage pass/fail. Coverage
and the block score are not a learned confidence. Zero disparity can be valid
when configured bounds include zero; it cannot yield finite physical depth.
The existing calibrated Q reprojection performs its own finite/physical limits.
Empty selections emit current empty coverage without model I/O.

## Runtime and display

`info` records original/network/head geometry, conversion factor, masks/coverage,
pair evidence and artifact identity. Preview is a uint8 diagnostic with percentile
or fixed display bounds. Its title states source pixels. Diagnostic stretching
never changes disparity. Lean or disabled result graphics omits Preview/graphics;
the numerical map and masks remain. The filmstrip override can select Preview.

The common DNN layer owns engine/target selection, model caches, explicit fallback,
sealed model/configuration/preprocessing verification and optional detailed
receipts. CPU is qualified here. The pinned ETH3D export needs New on this build;
Classic rejects its GatherND import. The example requests New explicitly and
does not retry another engine silently. Receipt selection is distinct from proof
of per-layer execution placement.

Source pixels, the combined two-view input elements, head dimensions and total
output elements are bounded. Model bytes are bounded and snapshotted once per
activation. Exceptions reset native state and the base executor contains them
without publishing a new successful result. No download or training occurs while
executing a block. Sealed Save As copies exact model bytes and notices.

## Evidence and scope

Original paired ramp controls exercise both bindings, channel packing, layouts,
unit/sign/affine conversion, low-resolution heads, rounded letterbox, fractional
right-mask support, tickets and known Q camera rays. The physical control has
fx=fy=80, a 0.1 m baseline and disparity 3 pixels: depth is 8/3 m, with independently
computed XYZ on the original source grid.

The native projects use a CC BY 4.0 T-LESS image resized to 320x240 and shifted
copies, with known artificial disparities 0, 8 and 16 pixels. These are not real
stereo acquisitions or measured depths of those photographed parts. On a fixed
interior, direct native inference has mean absolute errors **0.20038**, **0.03025**
and **0.03049 source pixels**, respectively. The 8/16-pixel controls retain 100%
of evaluation support and have no errors above one pixel there. The zero plane
retains **94.12%** under nonnegative-contributor and correspondence gating; small
negative predictions are invalidated, not clipped. The benchmark reports full
and retained-region errors separately. It does not establish performance on
occluded parts, another model, physical cameras or an accelerator.

See [sources, model member and hashes](<../examples/opencv tools/learned_stereo/SOURCES.md>)
and the independent `benchmark.py`/`benchmark.json` for reproducible evidence.
