# Brick edges

Open [brick_edges.json](brick_edges.json) and press **Run once**. The graph loads
a grayscale brick-wall photograph, smooths it with Gaussian blur, and uses Canny
to find edges. Select **Find mortar edges** to see the binary edge image. Adjust
the low and high thresholds, apply the settings, and rerun to compare the
mortar lines and the surface detail.

Run the graph headlessly from the repository root:

```powershell
opencv-dataflow-run examples/public/brick_edges/brick_edges.json
```
