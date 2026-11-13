# Change a selected surface's color

Open [masked_color_change.json](masked_color_change.json) and run once. Edit the
shared region or the red, green and blue multipliers, then inspect the image,
gradient/selection masks, maximum-channel change map and structured report.

The **Change selected color** block wraps native `cv2.colorChange`. Each
multiplier ranges from **0.5 to 2.5**. It scales channel gradients and solves
against the source boundary; it is not ordinary pixel multiplication or a
reference-image palette transfer.

## Independent reference

This original fixture places a BGR `[80,110,150]` rectangle on `[40,40,40]`.
With blue=0.7, green=1 and red=1.4, the independent interior reference is
`background + multiplier × (surface - background)` = **`[68,110,194]`**.
The native result matches those interior values on the qualified build, and the
vertical boundaries remain at **x=50 and x=142**. The limits and measurement
region are saved in [truth.json](truth.json). This simple reference does not
predict arbitrary gradient-domain edits on complicated scenes.

Use uint8 BGR/RGB, optionally with straight alpha, at least 8×8 and up to
2.1 million pixels. Channel order, alpha, source coordinates, tickets and
attached data survive. A mask and region are intersected before the native
solve. Only selected pixels are written back; surroundings and holes stay exact.
An empty selection or one erased by the native three-pixel gradient erosion is
an explicit no-op. UInt16/float images require an explicit conversion first.

The [masked photo example](../masked_photo_correction/README.md) explains the
selection versus native gradient masks. These tools edit appearance; use
white-balance or calibrated color tools for their corresponding measurements.

## Reproduce

[build_example.py](build_example.py) generates the original MIT image, graph
and reference. No customer or third-party imagery was used.

```powershell
python "examples/opencv tools/masked_color_change/build_example.py"
opencv-dataflow-run "examples/opencv tools/masked_color_change/masked_color_change.json" --profile debug
```

[Native API](https://docs.opencv.org/4.x/df/da0/group__photo__clone.html) ·
[OpenCV examples](../README.md)
