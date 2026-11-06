# Route image channels

Open [route_channels.json](route_channels.json) with **File → Open…** and
click **Run once**. **Extract image channel** takes the green plane from a
synthetic BGR image. **Route image channels** builds an RGB image with
the original red plane as R, zero as G, and the extracted green plane as B.

Find **Route image channels** under **Image processing → Channels**.
Select the output encoding and enter one source index for each output
channel in stored order. Source indices start at zero and run through
Image A before Image B. With three channels on A, index 3 names the
first channel on B. The example uses `[2, -1, 3]`. Use `-1` for zero
fill or repeat an index to duplicate a channel. An empty list selects
the first channels in order. A wired map overrides the entered list.
The block copies raw samples and tags them with the chosen output encoding.
[OpenCV channel routing reference](https://docs.opencv.org/5.0/main_modules/core_array.html#mixchannels)

Both inputs must have the same dimensions and pixel depth. **Same frame**
also checks their known spatial frames. **Pixel grid** pairs equal-size
images by pixel index. If both inputs carry tickets, their identities
must match. The output uses Image A's coordinates. Optional mask and
region inputs leave the output zero outside their intersection.

```powershell
opencv-dataflow-run examples/image_processing/route_channels/route_channels.json
```

[All examples](../../README.md)
