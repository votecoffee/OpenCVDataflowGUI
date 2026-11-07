# Check stabilization on recorded depth variation

Open [recorded_stabilization_validation.json](recorded_stabilization_validation.json)
with **File → Open…** and run all eight frames. Two editable rectangles select
areas of the recorded scene. The far patch supplies the fit and the near patch
checks whether that fit holds elsewhere in the image.

| Reference frame | Far patch median depth | Near patch median depth |
| --- | ---: | ---: |
| Anchor | 1.097 m | 0.833 m |
| Sixth frame | 1.294 m | 0.930 m |

The graph's **Far and near center depths** block shows the two center-point
readings on the anchor depth image. The table uses all valid samples in each
rectangle.

On the sixth frame, **Fit from the far patch** still accepts all of its
background tracks and covers about 98.7% of the image. In **Require near-patch
agreement**, 60 check tracks survive, but only 6 are within the 2.5-pixel
tolerance. Their median residual is **3.90 pixels**. The checked branch
withholds the image, coverage, and transform, then resumes on frame seven
with a new anchor. **Status → validation** reports the track counts and
residuals. The saved check requires 75% of surviving tracks to meet the limit.

```powershell
opencv-dataflow-run examples/flow_control/recorded_stabilization_validation/recorded_stabilization_validation.json --all-images
```

## Fixture provenance

The source is **TUM RGB-D benchmark, freiburg1_xyz**, by J. Sturm, N. Engelhard,
F. Endres, W. Burgard, and D. Cremers, *A Benchmark for the Evaluation of
RGB-D SLAM Systems* (IROS 2012). The fixture assets are available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Dataset and license](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)

The filenames retain the source timestamps. Eight consecutive RGB frames
from 1305031106.975547 to 1305031107.211358 were resized from 640 × 480 to
320 × 240 using area averaging and converted to 8-bit grayscale. The two
closest registered depth frames, 8.3 ms and 13.0 ms from the corresponding
RGB exposures, were sampled at `[1::2, 1::2]` to match the reduced grid. The
16-bit depth samples retain the dataset's scale of 5,000 samples per meter.
Zero samples are treated as missing. [TUM file formats](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)

The source archive SHA-256 is
`a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`.
[Source archive](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz)

[All examples](../../README.md)
