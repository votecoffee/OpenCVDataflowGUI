# Lab color classification

Open [lab_color_classification.json](lab_color_classification.json) with
**File → Open…**, then click **Run once**. The graph measures a patch
inside the blue disk and compares its mean Lab value with three named
references. The accepted label is **blue disk** and its ΔE is zero.
The preview shows the measured color on the left and the closest
reference on the right.

Set **References** to a JSON object mapping names to Lab triplets, such as
`{"dark gray": [30, 0, 0], "light gray": [70, 0, 0]}`. A wired palette on
the optional `references` port takes precedence. Choose CIE76 or CIEDE2000.
[CIEDE2000 implementation notes](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/)

**Maximum delta E** limits the nearest-reference distance.
**Minimum margin** sets the required ΔE gap between the two best references.
A single reference only needs to meet the distance limit. Set the margin to
zero to allow equal distances, with ties resolved in label order.
An out-of-tolerance or ambiguous sample emits no **Label**. The
**Classification** output still includes the closest candidate, margin,
and complete ranking.

```powershell
opencv-dataflow-run examples/lab_color_classification/lab_color_classification.json
```

[All examples](../README.md)
