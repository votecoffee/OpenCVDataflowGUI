# Sparse RLOF point tracking

Open [translated.json](translated.json) and select **Run Once**. The source
corner detector supplies 36 ordered points to **Track wired points (Sparse
RLOF)**. The second image moves the plate **5 pixels right and 3 pixels up**.
All 36 tracks pass. The reported median displacement is **[5.00, −3.00] px**.
Select **View accepted correspondences** to see the pairing between the two
images.

Open [occluded.json](occluded.json) to test rejection. The same motion is present
behind a flat rectangle. The round-trip and motion limits leave **20 of 36**
tracks, below the configured minimum of **28**. The inspection fails with
`sparse_rlof_insufficient_tracks` and publishes the surviving matches and the
reason for each rejected point. The original synthetic PNGs are covered by the
repository license.

## Inputs and outputs

- Wire source and target images of the same size and an ordered list of source
  `[x,y]` pixels, for example the **Good features / corners** point output.
  The source mask and region can limit which input points are tracked.
- Choose fixed intensity windows or cross-shaped color support. The block
  converts the working images to 8-bit and leaves the preview image in its
  source format. Wire target-grid initial guesses or enable the global motion
  prior. Guesses follow the source point order.
- Use **FB tolerance** to limit forward/backward round-trip distance in
  source pixels. With this check disabled, per-track `fb_error` is null.
  **Max displacement** independently bounds motion magnitude.
- **Target points** carries the target frame and ticket. **Matches** carries
  both image contexts and can feed match review or correspondence-based
  registration. **Tracks** retains one row per input with displacement, status,
  rejection reason and round-trip error when measured. Green arrows show
  accepted matches and red markers show rejected source points on the preview.

The block uses [OpenCV sparse RLOF](https://docs.opencv.org/5.0/extra_modules/optflow.html).
Its images are limited to 2,097,152 pixels each and its input to 10,000 points.
The fixture verifies local motion and rejection on a controlled pair. Round-trip
agreement is a consistency check. Repeated textures can still produce plausible
matches at the wrong location.
