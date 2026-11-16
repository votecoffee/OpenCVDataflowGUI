# Restore detail from registered neighbors

`sequence.restore_neighbors` restores one blurred capture from sharper images in
an ordered batch. It belongs to **OpenCV > Video Analysis (video)**. Use upstream registration or
trajectory replay first; this tool does not estimate motion. Open the
[six examples](<../examples/opencv tools/neighbor_restoration/README.md>) to try it.

## Inputs and blur model

Supply 2-16 equally sized and encoded gray, RGB, BGR, RGBA or BGRA captures at
uint8, uint16, float32 or float64 depth. Choose the zero-based target and temporal
radius. In **same_frame** mode every capture must have the same known spatial
root, root extents and pixel-to-root transform. **pixel_grid** explicitly asserts
that corresponding pixel indices have already been registered; it permits
independent acquisition roots. Registration accuracy remains the caller's job.

The expected target blur is a horizontal box, vertical box or square Gaussian.
Set its odd size and Gaussian sigma where applicable. The tool applies that
model to each donor and compares the prediction against the target, using the
maximum absolute difference across color channels. Integer signals share their
type's full scale; floating captures share one batch peak absolute value (or one
for an all-zero batch). Model-error tolerances use these scaled units, rather
than independently normalized images. Lighting changes can therefore reject a
donor. Alpha is excluded from intensity scoring and retained exactly from the
target; premultiplied input must first be unassociated. Changed alpha also rejects
transfer in its configured change halo.

Native 3x3 Sobel luminance energy, averaged over the selected odd energy window,
must strictly improve on target energy by the configured ratio and meet the
positive donor energy floor. Entire blur and scoring neighborhoods must be valid.
The support halo is `max(blur_size // 2, 1 + energy_window // 2)` pixels.
Model mismatches are dilated by **change_radius** before transfer is considered.

Use **best** to copy the highest-energy accepted donor, with earliest batch index
winning exact ties. A full mix copies donor sample bits exactly, including
Float64. **weighted** averages accepted donors by local energy. **mix** blends the
result with the target; computation uses bounded scaled Float64 values and integer
results are rounded to their original depth. No sharpening is invented where no
donor qualifies.

## Coverage, selection and identity

An optional same-order batch supplies uint8 coverage masks on each capture's own
grid. A mask carrying spatial metadata must match its corresponding capture,
including in pixel-grid mode. Masks without spatial metadata assert pixel-index
alignment. The target support is never expanded and invalid target samples are
copied unchanged. Transparent target pixels are unsupported. Donor holes and
image borders cannot contribute incomplete scoring or blur neighborhoods.

The optional mask and region select output centers on the target grid. They
intersect, and they do not crop the filtering neighborhoods. Outside selection,
outside original support, around detected changes and where no sharper donor
qualifies, target samples stay exact. Empty selections are invalid. A selection
with no supported pixels fails its inspection result.

Outputs retain the current target's ticket, spatial/pyramid coordinates and
custom payload. An outer batch ticket does not replace the target identity.
Neighbor tickets remain in the report as contributor provenance. No geometric
resampling or output coordinate change occurs here.

## Results and judgments

| Output | Meaning |
| --- | --- |
| `image`, `source` | Restored and original target images. |
| `valid_mask` | Original target support, intersected with nonzero target alpha. |
| `restored_mask` | Pixels changed by donor replacement or mixing. |
| `donor_index` | Strongest accepted original batch index, or -1 when unused. |
| `contributor_count` | Eligible donors per pixel; best mode uses one of them. |
| `contributions` | Actual contributor masks in batch order, all on the target grid. |
| `info` | Blur/scale/support details, restored fraction and per-neighbor eligible, changed and used counts with source tickets. |
| `preview` | Optional colored donor-index diagnostic. |

`minimum_restored_fraction` divides restored pixels by selected supported target
pixels. With **no_neighbor=pass_through**, a usable unchanged target can pass when
that minimum is zero; the report says `unchanged`. Choose **fail** to require at
least one restored pixel. These judgments describe restoration availability,
not whether the part passed inspection. Range, batch, format, coordinate and
nonfinite errors use the common block exception/validation framework. Native
filter exceptions are contained and a later activation can recover.

The block has no cursor or trained state, supports read-only concurrent use and
does not change OpenCV thread settings. A conservative 256 MiB workspace estimate
is checked before conversion or filtering. Optional maps/drawings respect the
global and per-block graphics settings; primary images, masks and reports still
work in Lean mode. The common executor owns timing, events and logging.

## Qualification and limits

The native C++ [WeightingDeblurer](https://github.com/opencv/opencv_contrib/blob/5.x/modules/videostab/src/deblurring.cpp)
is not exposed by the supported Python wheel. This block is a tested application
equivalent built from native filters, with explicit blur-model and change gates;
it does not claim numerical parity with that C++ implementation.

Independent finite box averages and analytic sharp patterns verify detail error
reduction and exact recovered samples. Controlled moved, appearing and disappearing
parts remain unchanged around detected differences. Saved projects cover an
alternating sharp/blurred/sharp sequence, Gaussian blur, 16-bit color with coverage
and region selection, unchanged pass-through and deliberate failure. Tests also
cover all supported depths/channel orders, extreme floating magnitudes, tickets,
support footprints, concurrency, XML/reopen, offscreen dropdowns and recovery.

The scene-consistency test is not an occlusion or defect oracle. Small,
low-contrast or model-indistinguishable changes can pass it; an inaccurate blur
model or registration may reject useful detail or admit incorrect detail.
Weighted fusion can soften edges. Qualify model/error/halo parameters on actual
captures, preserve raw target images and use them for critical defect decisions.
Blind blur estimation, arbitrary kernels and dynamic flow alignment are separate
capabilities. No camera or production acceptance is claimed by these fixtures.
