# Temporal non-local means

Open [temporal_nlm.json](temporal_nlm.json) with **File → Open…**.
Click **Run once** three times. The file source replays three noisy
views of a static scene. **Batch serial entries** assembles them in
order. **Temporal non-local means** then denoises the middle frame
and emits it with the ticket for `frame_02.png`.

The block is under **Image processing → Filters**. Connect an
ordered batch with matching image dimensions, depth, and channel
order. Use frames of the same scene on an aligned pixel grid.
Register moving frames before batching. The block rejects differing
transforms when frames declare the same source root.

**Target index** is zero-based. The default chooses the middle
frame, using the earlier of the two middle frames for an even-sized
batch. The odd **Temporal window** must fit on both sides of the
target. For each batch, the block emits a single image and reports
the window and target indices.

**Channels** mode supports 8-bit L1 or L2 denoising and 16-bit L1
denoising. **Luminance chroma** uses separate luminance and chroma
strengths on 8-bit color. Both modes preserve the target's alpha.
A mask or region limits where the denoised pixels are written.
[OpenCV denoising reference](https://docs.opencv.org/5.0/main_modules/photo_denoise.html)

```powershell
opencv-dataflow-run examples/image_processing/temporal_nlm/temporal_nlm.json --all-images
```

[All examples](../../README.md)
