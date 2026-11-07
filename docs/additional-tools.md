# Additional OpenCV blocks

These 12 eager OpenCV wrappers share the existing typed-port, result and inspector
contracts. They are available immediately in the toolbox after restarting the app.

| Type / subtype | Tool | Behavior |
| --- | --- | --- |
| Image processing / Contrast | Equalize histogram | `equalizeHist` on uint8 luminance; source-format or grayscale output |
| Image processing / Contrast | CLAHE local contrast | `createCLAHE().apply` on uint8/uint16 luminance; source-format or grayscale output, configurable clip limit and tile counts |
| Image processing / Contrast | Normalize intensity | Global `normalize(..., NORM_MINMAX)` with explicit output range and uint8/uint16/float32 depth |
| Image processing / Contrast | Gamma correction | uint8 `LUT`; `output=255*(input/255)^(1/gamma)`, alpha preserved |
| Image processing / Channels | Split three channels | `split` preserves the actual channel order, including RGB/HSV; three monochrome outputs |
| Image processing / Channels | Merge three channels | `merge` of identical-size/depth monochrome planes; choose BGR/RGB/HSV/Lab output tag |
| Image processing / Image math | Apply mask | `copyTo`; nonzero uint8 mask pixels copy the source, zero mask pixels produce zero output |
| Image processing / Image math | Absolute difference | `absdiff`; unsigned differences do not wrap, inputs must have matching size and depth |
| Inspection / Features | Good features / corners | `goodFeaturesToTrack`, Shi–Tomasi or Harris; corner list, count and overlay |
| Inspection / Features | Hough line segments | `HoughLinesP`; edge input, pixel line segments, count and overlay |
| Inspection / Features | Hough circles | Gradient and Gradient ALT, centers and radii, native votes, sortable circle collection and overlay |
| Image processing / Morphology | Distance transform | `distanceTransform` L2, approximate 3/5 mask or precise; raw float32 pixel distances and separate preview |

Corner lists and circle centers are lists of `[x,y]` pairs in the input image's
coordinate space, suitable for Pixel to physical when using a matching mapping.
Line segments are `[x1,y1,x2,y2]`, and circles are `[x,y,radius]`. No detection is a
valid empty result rather than an execution error. These tools do not automatically
choose application-level acceptance limits or guarantee detection on arbitrary images.

Normalization operates on the entire array, not independently per channel; a
constant image becomes the configured minimum. Set maximum to 1 for normalized
float display. Convert HSV/Lab to gray/BGR before normalization or gamma correction
to avoid treating encoded chroma/hue as brightness. Gamma requires uint8.
Equalization uses uint8 internally. CLAHE also works at uint16. Source-format
contrast output restores the original encoding and depth after processing.
Corners accept uint8/float32; Hough tools and
distance transform require uint8. Color-capable detectors use canonical grayscale.
Channel merge supports uint8/uint16/float32, with HSV/Lab limited to uint8/float32.

Distance transform treats zero pixels as background and reports distance from
foreground to the nearest zero in **pixels**, not physical depth or calibrated
millimeters. An all-foreground image is rejected because it has no background
reference. Its normalized preview must not be used for distance measurement.

## Lena feature project

First fetch the optional image with `python scripts/prepare_lena.py`. Open
`examples/demos/lena_features/lena_features.json`, then press F5. The graph contains:

```text
Lena -> Grayscale -> CLAHE -> Corners
                        | -> Histogram
                        + -> Canny -> Hough line segments
```

Try changing the CLAHE clip limit, corner quality/minimum distance, or Hough votes/
minimum length. Apply configuration and rerun. The input is project-relative and
the graph has no output writers, so repeated runs do not alter the image or write
files. Detector counts vary with parameters and OpenCV versions; automated checks
assert valid results and nonempty default detections instead of arbitrary exact counts.

Headless check:

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/demos/lena_features/lena_features.json --repeat 3
```

Node positions are stored separately from connections in every saved graph and
nested ToolBlock. Moving nodes marks the job dirty without clearing results.
Use Save/Ctrl+S to persist the rearrangement; zoom/pan are session view controls.

References: [OpenCV feature detection](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html),
[histogram equalization](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html),
[distance transforms](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html).
