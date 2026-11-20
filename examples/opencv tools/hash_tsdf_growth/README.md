# Compare bounded and seeded HashTSDF growth

Open [hash_tsdf_growth.json](hash_tsdf_growth.json) and choose **Run All**.
The batch collects six views of a synthetic plane at world Z = 1.009 m.
Poses advance laterally and closer to the plane, so each depth frame
observes a new region. The first five runs fill the batch. The last run
reconstructs both branches.

| Branch | Reserved blocks | Surface extent on X |
| --- | ---: | --- |
| Grow across all six views | 180 | −0.36 to 2.22 m |
| Keep first view's allocation | 60 | −0.36 to 0.42 m |

The growing branch stops at the budget set by **Block limit** and
**Maximum workspace MiB**. To see a refusal, set its block limit to 160
and rerun the project. The sixth view is rejected before native
integration. The report records the reason and the cloud retains the
surface admitted through the fifth view.

A frame's conservative reservation encloses the measured depth rays
and their truncation bands. The report shows the reservation count,
overlap with earlier reserved blocks, whether growth was enabled and
the native volume bounds. The raw native counts are diagnostic. The
OpenCV 5.0.0 CPU build used here reports both native counters as one
throughout the sweep. The memory estimate includes the native pool,
depth work buffers and an allowance for extraction.

The cloud keeps extracted points that agree with an admitted camera
depth within **Surface tolerance m**. This filters false crossings near
block boundaries. The fixture recovers unit normals and keeps every
emitted point within 26 mm axial error of the known plane.

For a headless run, use `opencv-dataflow-run "examples/opencv tools/hash_tsdf_growth/hash_tsdf_growth.json" --repeat 6`.
[OpenCV Volume reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1Volume.html)

The recordings and [builder](build_example.py) are original
OpenCVDataflowGUI contributions under the repository's MIT license.
