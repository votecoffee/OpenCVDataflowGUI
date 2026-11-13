# Sequence memory and warmup

Open [sequence_memory.json](sequence_memory.json). The source supplies synthetic
moving frames, so no camera or recording is needed.

1. Run once, then continue stepping. Compare **Running background subtraction**,
   **Farneback optical flow**, **Track points (Lucas-Kanade)** and
   **Image memory / trained reference**.
2. Observe which blocks need an earlier frame or warmup before emitting their
   first result. A waiting tool has not necessarily failed.
3. Use **Continuous** for a longer run. Inspect how retained state changes later
   results, then stop. For a bounded headless check, use `--repeat 8`.

This tour demonstrates the runtime's sequence/state behavior. Dedicated optical
flow and tracking algorithms are documented in the OpenCV tools collection.
Next: [inspect native 3D data](../rgbd_cloud/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)
