# Descriptor radius matching

Open [descriptor_radius.json](descriptor_radius.json) with **File → Open…**
and click **Run once**. The graph detects BRISK features in a synthetic texture
and its shifted copy, then passes up to two candidates per query keypoint to
**Estimate image homography**. The bundled fixture returned 416 candidates and
332 inliers, recovering about (9.01, 13.11) pixels of the known (9, 13) pixel shift.

**Match descriptors within radius** is under **Inspection → Matching**.
**Max distance** is inclusive and zero selects identical descriptors.
**Max neighbors** caps candidates per query keypoint, ordered by distance.
The matcher uses Hamming for BRISK and binary AKAZE, Hamming or Hamming2 for
ORB depending on WTA_K, and L2 for SIFT, KAZE, and float AKAZE. Choose a
distance appropriate for the descriptor family. The two feature sets must
have compatible layouts. [OpenCV descriptor matching reference](https://docs.opencv.org/5.0/javadoc/org/opencv/features/DescriptorMatcher.html)

The search compares every query descriptor to every train descriptor. It
accepts up to 10,000 descriptors per input and 10 million comparisons per
execution. If a graph exceeds either limit, reduce the detector's feature cap.

```powershell
opencv-dataflow-run "examples/opencv tools/descriptor_radius/descriptor_radius.json"
```

[All examples](../../README.md)
