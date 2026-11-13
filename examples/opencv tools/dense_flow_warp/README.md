# Warp a frame with a dense motion field

Open [dense_flow_warp.json](dense_flow_warp.json), choose **Debug**, and run once.
The original 16-bit RGB fixture moves **+5 pixels right and +3 down** onto an
explicit target grid. Its **11,439 covered pixels** match the independently
constructed target samples exactly. Select the warp block to inspect the image,
coverage and diagnostic outputs. Double-click it for the expanded editor.

The graph uses **Load dense flow file** to read original `.flo` fixtures.
The warp also accepts existing Farneback/DIS/LK `DenseFlow` outputs or floating
H×W×2 arrays from a user block. Files store vectors, not source/target identities;
the report labels their grid association as `size_only`. Pair them explicitly.

## Choose the direction

- **target_to_source**: the field is indexed on the output grid. For output
  `(x,y)`, sample the source at `(x+dx, y+dy)`. A previous-to-current optical flow
  can pull the current image back onto the previous image's grid.
- **source_to_target**: the field is indexed on the source grid. The block solves
  `source_point + flow(source_point) = target_point` by bounded fixed-point
  iteration. Inspect **Inverse error** and set its tolerance and iteration budget.
  Nonconvergence loses coverage. Arbitrary folds or global invertibility are not established.

The source, flow and target dimensions must match. A known flow grid must agree
with its declared source/output frame; anonymous fields are checked by size.
An optional **Target** image supplies output coordinates and pyramid metadata;
its pixels are unused. Without an anchor, a known pull-flow grid is retained or
a distinct output frame is created. The source image's ticket, acquisition time
and companion data remain primary. The report records source/target lineage.
No homography is invented for a nonlinear warp: exported donor maps carry the
explicit source/output grids and can feed the generic Remap block.

## Read the support masks

| Output | Meaning |
| --- | --- |
| `coverage` | Valid image-kernel donors that pass the enabled checks. |
| `invalid` | Nonfinite/oversized or unsupported flow, inverse failure, or unevaluable reverse evidence. |
| `uncovered` | Out-of-frame or unsupported source donors, including source selections. |
| `occluded` | Round-trip disagreement **proxy**, available only when reverse checking runs. |
| `roundtrip_error` | Pixel return error; NaN means the test was not evaluable at that pixel. |
| `inverse_error` | Forward-field inversion residual; unavailable in direct pull mode. |
| `map_x`, `map_y` | Float32 destination-to-source donor coordinates; use with coverage. |
| `coverage_preview` | Green valid, red invalid, gray unavailable, amber inconsistent. |

The four classification masks partition the output when reverse checking is
enabled. Without it, `occluded` emits no value and the report says `not_checked`.
Disagreement can also come from a bad flow estimate; it does not prove physical
occlusion or collision-free geometry. Reverse support masks can distinguish
unknown flow from measured disagreement.

**Mask** and **Region** refer to source donor support. Their intersection moves
with the samples; it is not an output-space write selection. Linear sampling
requires supported contributing donors. Cubic/Lanczos use conservative 2/4-pixel
halos. Native extrapolated borders never become valid coverage. Default invalid
fill is the constant border value; `native_border` keeps extrapolated samples
for display while the masks still identify them as unavailable.

## Other projects

- [invalid_and_occlusion_proxy.json](invalid_and_occlusion_proxy.json) separates
  **20 nonfinite-flow pixels**, **216 deliberately inconsistent pixels**, and
  **849 uncovered pixels**. The remaining **11,203** samples are correct. These
  are controlled field defects, not captured physical-occlusion ground truth.
- [coverage_gate_failure.json](coverage_gate_failure.json) requires 99% coverage
  but only gets about 93.1%. It fails and withholds the image while retaining
  masks, donor maps and the report. The headless runner intentionally exits 1.

## Controls and limits

Nearest, linear, cubic and Lanczos sampling and the native border modes use
dropdowns. Color order and sample depth survive, including resampled alpha.
Supported finite image samples are uint8, uint16, int16, float32 and float64;
physical depth planes and int32 labels require their corresponding tools or
an explicit conversion. Float64 cubic is rejected because of the qualified
native boundary-corruption case; use linear or float32 instead.

Images/fields are limited to 2.1 million pixels and fewer than 32767 pixels per
axis. Maximum motion is a per-axis validity limit. Work and inversion settings
bound the iteration cost. Border values must fit the stored sample range;
channel lists follow the input's stored order.
Zero coverage always fails; **Minimum coverage** can impose a stronger gate.
Lean skips the optional color preview unless its output is explicitly exposed.
The masks and processing result remain available independently of graphics.

## Source and reproduction

All images, fields, graphs and [truth.json](truth.json) are original project
fixtures under MIT. No customer or upstream imagery is included. The target
is made by array slicing, independently of the warp implementation.
[build_example.py](build_example.py) regenerates all three projects.

```powershell
python "examples/opencv tools/dense_flow_warp/build_example.py"
opencv-dataflow-run "examples/opencv tools/dense_flow_warp/dense_flow_warp.json" --profile debug
```

[OpenCV remap](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)
documents destination-to-source sampling and supported native controls.
[All OpenCV examples](../README.md)
