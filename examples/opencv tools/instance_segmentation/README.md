# Instance segmentation controls

Open `roi.json` and select **Run once**. No plugin, download or optional runtime
package is required. Inference produces detector rows and masks; the decoder
shows separate per-instance regions, a union mask, an overlap-count image and a
summary label image. The original image reaches the filmstrip with drawn outlines,
holes and ID/class/confidence/area labels in Debug. Lean skips optional graphics.

The bundled original ONNX model has fixed box proposals and mask templates, with
scores and mask gain driven by the current RGB image means. It uses explicit
tensor tiling to avoid an importer broadcasting failure on the supported wheel.
It contains no trained weights. These projects demonstrate numerical decoding and
application wiring, not learned part recognition. See [contracts](../../../docs/instance-segmentation.md).

| Project | Demonstrates |
| --- | --- |
| [roi](roi.json) | One mask per raw detection; duplicate row suppression preserves original mask identity |
| [letterbox](letterbox.json) | Rounded 80x48 -> 32x19 content with top/bottom padding 6/7; clipped ROI masks keep original box support |
| [per_class](per_class.json) | Class-indexed mask channels |
| [channels_last](channels_last.json) | Explicit NHWC ROI mask layout |
| [logits_after](logits_after.json) | Linear score sampling followed by sigmoid/threshold |
| [prototypes](prototypes.json) | Packed coefficients and shared prototype planes, cropped to retained boxes |
| [prototype_channels](prototype_channels.json) | Channel-first packed rows and channel-last prototype planes |
| [prototype_no_crop](prototype_no_crop.json) | Full-source prototype support with box cropping disabled |
| [separate_heads](separate_heads.json) | Pack separate XYXY boxes, scores and class IDs without losing ROI masks |
| [selected_roi](selected_roi.json) | Pixel ROI selection, sort by mask area, select an individual region/mask and find its blobs |
| [empty](empty.json) | Intentional normal failure: black input has no scored detections |
| [count_failure](count_failure.json) | Intentional normal failure: three valid instances do not meet minimum count four |

Summary labels resolve overlap by detector confidence; each item's own
`mask_region` still contains all of its predicted pixels. IDs remain original raw
row IDs after NMS/sorting; summary label IDs are raw ID + 1. To inspect an entire
instance, use **Select instance mask**, rather than extract its ID from the
summary label image. Mask area is available as the custom collection metric
`mask_area`; ordinary `area` remains box area.

`truth.json` contains independent reference masks as row-major packed bits in hex
(NumPy `packbits`, big bit order), expected scores/counts/judgments and asset
SHA-256 values. Decode a reference with `unpackbits(..., count=width*height)` and
reshape to `[height,width]`. The generator calculates image means, scalar mask
sampling, class/channel selection and box support without running inference or
the decoder to obtain its expected masks. Linear references qualify the supported
CPU float64 remapping path, including its 1/32-pixel phases. Other native sampler
implementations can differ near threshold boundaries.

Run in the project environment, for example:

```powershell
.\.venv\Scripts\python -m opencv_dataflow.runner 'examples/opencv tools/instance_segmentation/roi.json'
```

To regenerate, install the development-only `onnx` package and run
`build_example.py` with `PYTHONPATH=src`. Different existing assets are preserved
with an error. Automated regeneration builds in a fresh temporary folder, checks
all JSON definitions and exact asset hashes, and verifies the originals remain
unchanged. Save As rebases the ordinary source/model paths.

All images, model arithmetic, proposals/templates, graphs and numerical reference
calculations are original work by OpenCVDataflowGUI contributors under this
repository's [MIT license](../../../LICENSE). Each graph includes an on-canvas
attribution comment. No customer images or third-party models/data are included.
