# PCB outline

Open [pcb_outline.json](pcb_outline.json). Press **Run once** for the first
photo and again for the second. Select **Draw outline on photo** to see the
contour on the original color image. Use **Reset flow state** to replay the
two-photo sequence.

The graph converts the photo to HSV, selects the blue board, closes small
gaps in the mask, and checks that exactly one external contour exceeds the
minimum area. This example checks board presence and extracts the outline.
Measurements are in pixels. Both photos are normal samples from VisA. See
[image credits and license](../assets/README.md#visual-anomaly-visa-pcb-photographs).

To run both photos headlessly from the repository root:

```powershell
opencv-dataflow-run examples/public/pcb_outline/pcb_outline.json --all-images
```
