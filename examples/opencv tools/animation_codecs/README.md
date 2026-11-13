# Animated image codecs

Open a project below and use **Run once**. Tools appear under **OpenCV → Image
Codecs → Codecs**; the file source is in **Files**. These original MIT fixtures
demonstrate native APNG, GIF and WebP file/buffer APIs without external plugins.

| Project | Expected result |
| --- | --- |
| [APNG round trip](apng_roundtrip.json) | Three frames retain their order, 80/120/200 ms durations, and loop count 3. Image norm is exactly zero. |
| [Lossless WebP round trip](webp_roundtrip.json) | The same timing and exact visible samples, with per-frame context preserved. |
| [GIF timing and palette](gif_timing.json) | Explicitly floors 83/129/208 ms to 80/120/200 ms. Inspect both lists in Info and the decoded color reduction. |
| [Exact GIF timing rejection](gif_timing_failure.json) | An intentional validation error at Encode animation. Decode stays inactive and no encoded output is emitted. |
| [Inspect every frame](frame_inspection.json) | Unbatch sends all three images through thresholding and component inspection. Each has one circle; the last result has centroid (116,50). |
| [Write verified WebP](write_webp.json) | Writes `animation-output.webp` beside the project. Choose a fresh filename for subsequent runs; existing files are preserved. This generated output is ignored by Git. |

The node image shows the selected preview frame, with optional drawable frame
index/timing graphics. `Frames` carries the complete ordered group; use **Select
group entry** to choose one, or **Unbatch serial entries** to inspect every frame.
Use Results → Info to read frame counts, durations, loop metadata and sample errors.
These blocks decode a bounded group in one activation; they do not pace GUI playback.

One file acquisition ticket owns the loaded group. Each image also carries its
frame index and duration. Encoding an existing group preserves its individual
frame tickets, coordinate frames and companion payloads through the memory packet.
Logical RGB/RGBA order is restored when decoding that packet with **Unchanged**;
file/raw-buffer decoding uses OpenCV's canonical channel order.

[truth.json](truth.json) records the independent design geometry and timing.
The three [PNG reference frames](assets/frame_0.png) and animated
[APNG input](assets/moving_part.png) are generated from the same fixed pixel formula.
Regenerate the projects and fixtures from the repository environment:

```powershell
python "examples/opencv tools/animation_codecs/build_example.py"
```

APNG and WebP may merge identical frames; the blocks reject a changed count or
timing before publishing it. GIF is palette based and uses binary transparency.
AVIF is unavailable in the current native build. Separate APNG posters are
qualified for grayscale/BGRA; a native RGB-poster defect is rejected. See the
[animation contract](../../../docs/animation-codecs.md) for bounds, controls and
backend limitations. All images, projects and builder code in this folder are
original work under the repository's MIT license; there is no third-party imagery.
