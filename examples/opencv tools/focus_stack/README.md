# Focus stacking and focus-level selection

Open `project.json` and run once. Two 16-bit RGB captures show complementary
sharp parts of an original resolution coupon. **Focus stack images** chooses
sharp pixels and preserves the anchor in ambiguous areas. The selected interior
must exactly match the independent sharp reference: maximum error falls from
19,912 counts before fusion to zero afterward.

`registered.json` first registers a capture shifted by `[2, -1]` pixels using the
ordinary **Align images (ECC)** block. Its shared sharp registration region
produces the expected `[-2, 1]` source-to-target translation. Nearest sampling
preserves the fixture's original pixels. The same independent comparison then
verifies the fused result. The evaluation region excludes resampling gaps and
scoring halos. Full-field ECC can be biased by changing focus; a good correlation
coefficient alone does not establish pixel alignment.

The overlapping equally sharp area intentionally reports no unique focus level.
Its anchor pixels are already sharp, so the whole evaluation region still
matches truth. These projects do not treat ambiguous pixels as known depth.

## Inputs and ownership

`cv.focus_stack` accepts an ordered **FlowGroup** of 2–16 images from **Batch
serial entries**. Captures must share size, dtype and gray/RGB(A)/BGR(A) channel
order. Use fixed exposure, gain and encoding; changes in contrast or noise can
affect focus energy. The block performs fusion after registration. Ordinary
registration and warp blocks remain visible upstream in the graph.

**Same frame** checks known coordinate grids. **Pixel grid** explicitly declares
that equal pixel indices are already registered, even if their source frames
differ. It performs no additional alignment. A batch ticket owns the result when
provided; otherwise the anchor's identity is retained. Output coordinates,
pyramid context and companion payload come from the anchor. Info retains every
capture's original identity and coordinate context. Measurements and indices
remain in the current image/pyramid grid.

Optional **Coverage** is a same-order batch of uint8 masks describing captured
support after registration. Masks are paired with captures by index, and known
mask/capture frames must agree. Supply them for resampling gaps, occlusion masks
or other unusable samples. The optional ordinary mask and region select output
centers on the anchor grid; they do not cut scoring neighborhoods.

## Focus response and reliability

Choose local mean squared Sobel gradients (**Tenengrad**) or squared Laplacian.
The derivative aperture, odd averaging window, optional Gaussian prefilter,
sigma and border mode are configurable. Unsigned captures share integer full
scale; floating captures share one batch peak. Scores therefore use a common
intensity scale, without independently normalizing each capture's contrast.
Thresholds depend on these settings and are not universal sharpness units.

**Strict** support requires complete captured neighborhoods, including the
prefilter, derivative and energy window. **Extrapolate** permits the chosen
image-border padding. Internal coverage holes are excluded with their full halo
in both modes. Transparent samples cannot provide scoring neighborhoods. Alpha
does not influence focus energy; output alpha is retained from the anchor.
Premultiplied inputs must be unassociated first.

Reliability requires enough contributing captures, positive energy at or above
the energy floor and sufficient `(best - second) / best` contrast. This contrast
is not a probability. If one contributor is explicitly allowed, its contrast is
reported as one, meaning uncontested evidence. Exact ties prefer the anchor,
then batch order. The default contrast threshold rejects ties. Selecting zero
permits them. Unreliable and unselected pixels retain the anchor unchanged.

The block passes when at least one pixel is reliable and the selected reliable
fraction meets **Minimum valid fraction**. Otherwise it returns a failed result
with the anchor fallback and explicit coverage diagnostics.

## Outputs and blending

- **Image** preserves dtype, channel order and anchor alpha. **Hard** selection
  copies winning pixels exactly. **Weighted** fusion uses relative energy raised
  to **Blend power**, with optional spatial weight smoothing. It is a bounded
  convex mixture and can soften details; it does not guarantee an all-sharp image.
- **Focus index** is the original zero-based batch index of the strongest focus
  response, or `-1` where fusion is unreliable/outside selection. In Weighted
  mode it names the strongest response, not every contributing source.
- **Valid mask**, **Confidence**, **Contributor count** and **Response** expose
  reliability, competing energy, supported capture count and best energy.
  Rejection counts in Info can overlap; they are diagnostic conditions.
- **Source** retains the unchanged anchor capture. The colored level preview is
  black where focus is undefined. A drawable coverage label belongs to Image.

Graphics-disabled runs skip colorization and drawing. Hard mode retains only
the leading scores during selection. Weighted mode caches score/support maps.
The stack is limited to 16,777,216 total image pixels, 64 million channel samples,
4,194,304 pixels per image and 16384 per axis. Complete output maps are retained.

Focus index and confidence are diagnostic image values. They are not calibrated
surface depth, part visibility, or an occlusion measurement. Focus breathing,
parallax, moving parts and sensor noise need application-specific qualification.

## Attribution and regeneration

All coupon images, reference values, projects and code are original work under
this repository's MIT license. `sharp_truth.png` is generated directly from the
resolution texture, holes and slot before any blur or fusion. The shifted image
and `truth.json` record the independent registration geometry.

```powershell
.\.venv\Scripts\python.exe "examples/opencv tools/focus_stack/generate.py"
```

Algorithm reference: [OpenCV image filtering documentation](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html).
