# Time-of-flight depth cleanup

Open [the project](tof_depth_cleanup.json) and run all three RGB-D frames.
The **Clean time-of-flight depth** block makes the original and cleaned depth
available for comparison. Its **Rejection reasons** preview shows amber for low
confidence, magenta for flying-pixel candidates, red for isolated outliers and
gray for holes already present in the input. The cleaned depth reconstructs as a
point cloud without the suspect samples.

The original synthetic recording is MIT-licensed with this repository. The
background step is 1.0 m to 1.6 m and moves one pixel per frame. A one-pixel-wide
feature stays at 0.75 m. Each frame includes 20 low-confidence samples,
5 intermediate edge samples, 3 isolated outliers and 6 missing samples.
The accepted 5,342 depths per frame are unchanged. Compare against the aligned
`truth_*.png` depth (millimeters) and `reasons_*.png` labels under `assets`.

| Code | Meaning |
| --- | --- |
| 0 | Unchanged or outside the selection |
| 1 | Low confidence |
| 2 | Flying-pixel candidate between supported near and far depths |
| 3 | Isolated outlier next to a coherent surface |
| 4 | Already invalid in the input |

The example uses synthetic artifacts. A single depth raster can reveal suspect
geometry, while determining multipath as a physical cause requires additional
sensor evidence. Agreement between nearby samples can hide coherent artifacts.
An isolated real feature can also resemble an outlier. Tune the physical
tolerances or disable a geometry check for those scenes.

Background on the artifact mechanisms is available in
[Mask-ToF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chugunov_Mask-ToF_Learning_Microlens_Masks_for_Flying_Pixel_Correction_in_Time-of-Flight_CVPR_2021_paper.pdf)
and [A Light Transport Model for Mitigating Multipath Interference in ToF Sensors](https://dspace.mit.edu/entities/publication/7d63918a-fa99-4782-8635-065115626d26).

Run the project headlessly.

```powershell
opencv-dataflow-run "examples/custom tools/tof_depth_cleanup/tof_depth_cleanup.json" --all-images
```

Run `build_example.py --output <empty-directory>` to regenerate the assets and
project in a new location.
