# Hit-or-miss binary patterns

**Hit-or-miss binary pattern** (`cv.hit_or_miss`) is under
**OpenCV → Image Processing → Morphology**. It matches an exact binary
neighborhood, such as a skeleton endpoint, corner, isolated pixel or junction.
Use thresholding and, when appropriate, thinning upstream. This is a local
pixel-pattern operation, not a scale/rotation-invariant object matcher.

## Kernel and input

Click **Edit kernel…** for the visual grid. In **Custom**, clicking a cell or
pressing Space cycles through:

| Value | Meaning | Color |
| --- | --- | --- |
| `0` | Ignore this cell | Unfilled |
| `1` | Require foreground | Blue |
| `-1` | Require background | Red |

Each cell also has a tooltip. Resizing and saved projects preserve negative
values. At least one cell must constrain foreground or background; an all-zero
kernel is invalid. Rectangle, ellipse, cross and diamond presets use foreground
constraints at their active cells. Ordinary Morphology retains its binary editor.

**Strict binary** accepts nonempty uint8 gray images containing 0/1 or 0/255.
They are normalized to 0/255 before matching. Mixed 1/255 foreground values,
unthresholded gray, color, floating-point and physical depth planes are rejected.
**Nonzero** explicitly treats every nonzero uint8 gray sample as foreground.
**Polarity → Dark** reverses foreground and background after normalization.

**Anchor** is the kernel position reported as the match coordinate; `-1` centers
that axis. Even and rectangular kernels and off-center anchors are supported.
**Shape anchor** is separate and only controls construction of a standard Cross.
**Iterations** is OpenCV's repeated erosion of each constraint set, followed by
their intersection; it does not repeatedly apply the whole hit-or-miss transform.

## Borders, regions and outputs

**Full kernel**, the default support policy, excludes centers whose complete
rectangular kernel footprint extends outside the image. This includes ignored
cells and the footprint expansion from Iterations. If no footprint fits, the
block returns an empty match mask without doing native morphology.
**Native border** retains matches from the selected OpenCV border extrapolation.
The default neutral border ignores missing erosion samples. Explicit `[0]` or
`[255]` border values apply separately to the original image and its complement;
they do not describe padding the original image once and then complementing it.

Optional **Mask** and **Region** inputs intersect and filter reported centers.
Neighborhoods still use the full source image. Matching known coordinate frames
are required; crop/transform a region or mask explicitly when its grid differs.

- **Image** is a complete uint8 0/255 match mask, with zero outside the selected
  centers and configured support. The source image is unchanged.
- **Count** is the total number of matching anchor pixels, not connected parts.
- **Points** contains `[x,y]` coordinates. **Max points** bounds memory by evenly
  sampling excess matches in row-major order. Zero omits coordinates. Sampling
  never truncates the mask, total count or inspection judgment.
- **Measurements** reports the total/returned counts, sampling flag, drawn count,
  effective anchor, kernel dimensions/constraint counts and support policy.
  `support_centers` counts the image centers allowed by that support policy,
  before the optional mask/region filter.
- Drawable result graphics mark returned points, green for a passing result or
  red for a count failure. **Max graphics** and the block/global graphics mode
  control their cost. The mask contains no baked-in markers.

**Check count** enables inclusive minimum/maximum limits using the complete
selected count. Maximum `-1` means no upper limit. A count outside the limits
emits a valid failed inspection with reason `hit_or_miss_count_out_of_range`.
Invalid settings/input or native exceptions are contained by the normal executor
and emit no processing result. Tickets, timing, coordinate/pyramid metadata and
other payloads follow the standard image/connection contracts.

Kernels are limited to 255×255; inputs to 16,384 per axis and 64 million pixels.
Large arbitrary kernels and many erosions can be expensive. This tool provides
exact local matching, not a calibrated confidence score.

[Six original examples](<../examples/opencv tools/hit_or_miss/README.md>) ·
[OpenCV image filtering API](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html) ·
[Native implementation](https://github.com/opencv/opencv/blob/5.x/modules/imgproc/src/morph.dispatch.cpp) ·
[Regions](regions.md) · [Block parameters](block-parameters.md)
