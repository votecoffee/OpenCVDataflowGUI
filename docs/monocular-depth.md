# Learned monocular depth

`ml.monocular_depth` runs a single-image ONNX depth head through the shared DNN
loader, sealed model contracts, CPU cache and execution receipts. The initial
qualified learned model is the unchanged MiDaS v2.1 Small release on OpenCV 5
Classic/CPU. See the [eleven projects](<../examples/opencv tools/monocular_depth/README.md>)
for exact arithmetic controls and a licensed real RGB-D capture.

## Quantities and ports

Choose the model's convention explicitly. Let `q = head_scale * head + head_offset`.
The fixed affine transformation happens before source-grid resampling. The block
never fits scale or offset from the current image or normalizes numerical output
per image.

| Convention | Estimate values | Physical Depth output |
| --- | --- | --- |
| relative_depth | Positive q, arbitrary units | No event |
| relative_inverse_depth | Positive q, larger usually means nearer | No event |
| metric_depth | Positive q in selected m or mm | z = q |
| metric_inverse_depth | Positive q in reciprocal selected units | z = 1/q, after interpolation |

`estimate` is a JSON data record containing source size, flattened row-major q,
units, convention, artifact identity and validity. Invalid entries are zero.
It deliberately uses a data port, so an arbitrary relative map cannot be wired
as physical depth. `depth` emits a `DepthArray` only for an explicitly declared
metric convention, with invalid entries NaN. Its m/mm tag is understood by the
existing point-cloud tools. Model-value and physical-depth bounds invalidate
samples rather than clipping them. Depth bounds use the selected physical units.

Selecting a metric convention is a caller declaration, not a calibration check.
Records and payloads state `metric_declared` and `metric_validated: false`.
The synthetic models verify conversion and camera-ray geometry; MiDaS remains
relative and cannot supply measured XYZ. A different trained metric model needs
its own scale, domain, geometry and held-out acquisition qualification.

`image` preserves original pixels, depth, alpha and encoding. `valid_mask` is a
source-sized uint8 selection/support map. `info` reports geometry, valid/selected
counts, quantity/depth/preview units and native-call count. `receipt` uses the
shared model/preprocessing/backend evidence. Ticket, recipe, frame and pyramid
context survive. Estimate, physical depth, mask and preview also carry a
`depth_estimation` provenance record; the original image's payload is unchanged.

## Preparation and geometry

Accepted sources are uint8, uint16 and finite float32 [0,1] ordinary gray,
RGB/BGR or alpha images. Integer full scale becomes unit float32. Conversion to
the configured model color order precedes resize, then per-channel mean/std
normalization and NCHW packing. Grayscale can be replicated into a color model;
alpha is excluded from inference. This does not alter the returned source.
Use zero mean/unit std for the pinned MiDaS ONNX: its RGB normalization is
already embedded in the graph. Do not apply it twice.

Preparation exposes stretch or rounded letterbox, linear/cubic/area input resize,
padding value, input/output names and network dimensions. One batch-one,
channel-one output is accepted in NHW, NCHW, NHWC, HW, CHW or HWC layout.
Nonfinite tensors or incompatible ranks are contained execution errors.

For source center x, network content width rw, left padding px, network width nw
and head width hw, the head sample coordinate is:

```
u = ((x + 0.5) * rw / source_width + px) * hw / nw - 0.5
```

The same rule applies vertically. The reported source-to-network matrix uses
pixel-center coordinates. A head cell must lie wholly inside unpadded content.
Linear sampling requires every nonzero-weight contributor to be supported;
even a tiny contribution from an invalid cell invalidates that output pixel.
Nearest sampling checks its selected cell. Boundary replication does not make
padded cells valid. OpenCV 5 float-map remapping is checked against an independent
full-float bilinear reference, rather than a separately quantized map table.

Mask and region intersect on the original source grid and gate output writes.
Excluded pixels remain model context. Empty selection emits current empty
coverage without loading a model. Minimum valid count/fraction gives an ordinary
inspection pass/fail, distinct from an exception. The score is selected-pixel
coverage, **not model confidence or depth accuracy**.

## Display and runtime

The uint8 diagnostic `preview` is either percentile stretched or displayed with
fixed bounds. Display scaling never affects q or physical depth. Result graphics
label relative arbitrary units or estimated physical units. Connect Preview to
the filmstrip override to choose that diagnostic for the final frame. Lean mode
or disabled result graphics omits preview/graphics; numerical processing remains.
Disable `export_estimate` to avoid the per-pixel JSON list when only typed metric
depth, coverage or display is needed.

Source pixels, configured input elements, head axes and output elements have
explicit guards. Raw models are bounded and snapshotted once per active execution;
sealed contracts also verify model bytes, configuration and preparation. Exceptions
reset native state and are contained by the base executor. No download, ground
truth lookup or calibration occurs during block execution. Accelerator controls
are exposed through the shared DNN interface, but this increment qualifies CPU.

## Independent evidence and limits

The arithmetic controls use a known RGB ramp and ONNX channel-selection head.
They check all layouts, positive/invalid values, affine/inverse conversion,
m/mm, rounded low-resolution letterbox, masks, holes, retained payloads, exact
point-cloud rays, Debug/Lean, save/reopen, sealed Save As and XML.

The native model runs on T-LESS scene 6/frame 2, a public 720x540 RGB-D capture
with 387,684 positive reference depth pixels. Ground truth is loaded separately
for evaluation and is never connected to inference. On 3,609 randomly selected
pixel pairs separated by at least 20 mm, near/far ordering accuracy is **97.20%**.
An evaluation-only inverse-depth scale/shift fit uses the left image half and
checks the right half of the same acquisition: RMSE **51.36 mm**, absolute
relative error **6.18%**. The constant calibration-median baseline has lower RMSE,
**40.70 mm**. These results do not validate metric inference or generalization to
another scene. The fit is recorded only in `benchmark.json`, never in runtime
parameters or outputs. Relative ordering is useful evidence; it does not establish
physical dimensions, robot clearance or safe picking.

The [source record](<../examples/opencv tools/monocular_depth/SOURCES.md>) includes
model/image hashes, MIT/CC BY 4.0 notices and unchanged upstream inputs. Model
weights are explicitly prepared into ignored `.local` bundles. No training
framework is needed to prepare or run the unchanged native release.
