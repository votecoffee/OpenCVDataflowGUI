# Sequence memory and warmup

Open [sequence_memory.json](sequence_memory.json). The source replays a bundled real
TUM camera recording, so no camera or additional download is needed.

1. Run once, then continue stepping. Compare **Running background subtraction**,
   **Farneback optical flow**, **Track points (Lucas-Kanade)** and
   **Image memory / trained reference**.
2. Observe which blocks need an earlier frame or warmup before emitting their
   first result. A waiting tool has not necessarily failed.
3. Use **Continuous** for a longer run. Inspect how retained state changes later
   results, then stop. For a bounded headless check, use `--repeat 8`. Camera motion contributes
   to the observed flow and background change; it is not object velocity.

This tour demonstrates the runtime's sequence/state behavior. Dedicated optical
flow and tracking algorithms are documented in the OpenCV tools collection.
Next: [inspect native 3D data](../rgbd_cloud/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)


## Source and scope

Real TUM RGB-D freiburg1_rpy frames (CC BY 4.0), using the already bundled
320×240 grayscale FFV1 excerpt. The demo replays twelve frames with their media timing.

[Full creator attribution, source hashes and changes](../../assets/showcase/SOURCES.md).
The original synthetic graph remains in [regression fixtures](../../assets/synthetic_graphs/README.md).
