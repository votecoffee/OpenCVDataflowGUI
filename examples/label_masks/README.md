# Masks from signed labels

Open [label_masks.json](label_masks.json) with **File → Open…** and
click **Run once**. Two seed regions initialize **Marker-controlled
watershed**. The watershed label image flows into two blocks.
**Compare pixels** selects label `-1`, which marks watershed boundaries.
**In range mask** selects labels `1` through `2`, which mark the two
region interiors.

Both blocks accept monochrome signed int32 label images from watershed,
labeled distance transform, or another segmentation tool. The resulting
uint8 masks retain the label image's coordinate frame. Optional mask and
region inputs restrict the selection.

**Compare pixels** can compare two label images or one label image and a
scalar. **In range mask** includes both bounds. Fractional bounds on an
int32 label plane select only the integers inside the requested interval.
[OpenCV comparison](https://docs.opencv.org/5.0/main_modules/core_array.html#compare) · [OpenCV range mask](https://docs.opencv.org/5.0/main_modules/core_array.html#inrange)

```powershell
opencv-dataflow-run examples/label_masks/label_masks.json
```

[All examples](../README.md)
