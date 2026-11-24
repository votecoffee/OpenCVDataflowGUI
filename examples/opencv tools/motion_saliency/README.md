# Motion saliency

Open [motion_saliency.json](motion_saliency.json) and choose **Run All**. The
five-second clip starts with a stationary scene. From 3.60 to 4.30 seconds a
bright part moves across the image. Compare the originals with outlines or
select **Mask** in the results. The filmstrip displays the Bin Wang mask.

| Observation on the example | Result |
| --- | --- |
| First mask after warmup | Frame 59 at 2.95 s |
| Bin Wang on the 15 moving frames | All 484 part pixels selected, with zero extra pixels |
| Retina at frame 77 | 0.415 IoU against the union of the current and preceding part outlines |
| Part disappears at frame 87 | Retina selects 487 transient pixels and clears on frame 88 |

**Motion saliency** is under **OpenCV → Saliency**. Use **Bin Wang** for a
background-based foreground mask or **Retina transient** for local changes
over time. Retina uses motion energy without per-frame normalization. Its
threshold and filter scales are configurable. Both methods process consecutive
frames in uint8 luminance and preserve the source coordinates and ticket.

Readiness and frame time appear in **Status** during warmup. Bin Wang
also waits to see more background than foreground before publishing its
first mask. Start with a steady scene. A **Reset** signal clears the current
stream, and a restart of the video frame index triggers fresh learning.
Bin Wang starts a new learning period when a pixel accumulates 190
foreground updates in a model generation. This bounds work before the
native neighborhood promotion step. The status reports the restart.
Masks and regions restrict the emitted pixels after full-frame learning.

The example clip is an original MIT asset. Rebuild from the repository root.

    python "examples/opencv tools/motion_saliency/build_example.py" --output "examples/opencv tools/motion_saliency"

[OpenCV Bin Wang motion saliency](https://docs.opencv.org/5.0/extra_modules/classcv_1_1saliency_1_1MotionSaliencyBinWangApr2014.html)
[OpenCV transient-area segmentation](https://docs.opencv.org/5.0/extra_modules/classcv_1_1bioinspired_1_1TransientAreasSegmentationModule.html)
