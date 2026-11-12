# Continuous centered camera stabilization

Open [continuous_camera_stabilization.json](continuous_camera_stabilization.json)
and choose **Run all**. The original lossless synthetic clip contains **144
frames**, combining slow camera travel with faster horizontal and vertical
jitter. **Stabilize continuous stream**, under **Custom → Sequence → Temporal
processing**, smooths the path with four future frames and a two-frame Gaussian
sigma. Open the source and corrected image popouts to inspect the result.

The first four activations wait for look-ahead. Later activations emit
individual corrected frames in source order. The final activation also emits
the remaining tail, so all 144 frames reach downstream consumers. Inspect
**Report** for observed and smoothed pose, smoothing-window bounds, coverage,
and segment information. **Buffered** and **Status** show readiness and
retained-state size.

- Image, Coverage, Transform and Report keep the processed frame's original
  ticket and video timestamp. The current triggering input can be several
  frames newer.
- Coverage marks supported samples. Transform contains raw-to-corrected and
  corrected-to-raw mappings, and both coordinate contexts. Pixel depth and
  color encoding are preserved.
- The window retains at most 2 × radius + 1 source images. The stored-state
  budget is checked before reserving the configured look-ahead. This budget
  excludes native intermediates, result snapshots and GUI copies.
- Look-ahead is a number of future arrivals. At 30 frames/s, four frames add
  about 133 ms of acquisition delay. If a live source pauses, pulse Flush to
  emit the tail using the available neighbors.
- Finite playback, ToolBlocks and acquisition queues flush automatically.
  Place a background acquisition queue before this block.
- Reset discards pending frames. Scene cuts and changed contexts flush the
  previous tail, then start a new coordinate segment. A coverage failure
  emits a failed Report for that frame and withholds its corrected image.

The automated example test verifies that the second-difference RMS of the
smoothed translation path is below 20% of the original path's value. Other
fixtures compare continuous output with the batch smoother, check native
pixel depths, and verify bounded state and ticketed tail delivery.

The clip, project and generator are original synthetic material under the
repository's MIT license. The project's layout comment records this attribution.
Run `python generate.py` here to recreate the FFV1 clip and graph.

From the repository root:

```powershell
opencv-dataflow-run "examples/custom tools/continuous_camera_stabilization/continuous_camera_stabilization.json" --all-images
```

See [sequence memory and buffered plugins](../../../docs/sequence-memory.md#continuous-centered-stabilization),
the [finite batch example](../offline_camera_stabilization/README.md), and
OpenCV's [tracking reference](https://docs.opencv.org/5.0/main_modules/video_track.html).

[All examples](../../README.md)
