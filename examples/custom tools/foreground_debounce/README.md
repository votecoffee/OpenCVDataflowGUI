# Debounce foreground components

Open `project.json` and **Run all**. It replays nine original foreground-label
images through **Custom > Sequence > Foreground masks > Debounce foreground
components**. The moving part crosses disjoint pixel locations; it must still
confirm after three consecutive observations. No camera or download is needed.

The first two activations show **Waiting** and emit no primary data. From
frame 2 onward, accepted component counts are **1, 1, 0, 0, 0, 1, 0**.
Glints at frames 1 and 3 never confirm. A missing part at frame 4 clears its
history; it confirms again only at frame 7. A stable 127-valued shadow is
excluded by the explicit threshold, and a one-pixel spike is below minimum area.
Accepted masks must match `assets/expected` exactly. `assets/truth.json`
records independently specified counts, positions and pixels.

The result graphics show confirmed boxes in green and pending boxes in amber,
with the current confirmation count. They are drawable objects that can be
overlaid on the original source image through the common graphics connections.
Expose **Rejected mask** or **Observations** to inspect current rejected
pixels and association diagnostics. Empty emitted masks are distinct from
warmup's absent events. Run all again, or use **File > Reset flow state**,
to begin a fresh sequence.

Association requires mutually unique centroid-distance, area-ratio and optional
bounding-box overlap gates. Ambiguous candidates restart confirmation. No
prediction, gap bridging or old pixels are emitted. Fast motion beyond the
configured gate, touching parts, merges, splits and crowded scenes can prevent
confirmation. Confirmation counts are not probabilities or reliable part IDs.

All images, expected masks, generator and graph are original content under the
repository's MIT license. Regenerate with
`python "examples/custom tools/foreground_debounce/generate.py"`.
See [sequence documentation](../../../docs/sequence-memory.md) and the
[OpenCV connected components reference](https://docs.opencv.org/5.0/main_modules/imgproc_shape.html).
