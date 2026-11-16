# Optical-flow files and image-pair association

**Load dense flow file** (`input.flow_file`) and **Save dense flow file**
(`output.flow_file`) transport Middlebury `.flo` fields through native
`readOpticalFlow` / `writeOpticalFlow`. The field is float32 `H×W×2`, with signed
horizontal dx and vertical dy in local pixels. These tools belong to **OpenCV →
Video Analysis → Motion fields**. Existing loader projects remain compatible.

## Read, write and inspect

The loader checks the signature, dimensions, exact file length and detected
size/mtime changes before native decoding. It probes a temporary snapshot through
Python file I/O, supporting Unicode source names on the installed Windows wheel.
The returned samples must match the validated file bytes. Files are limited to
**2.1 million vectors**, with each axis below 32767. Native staging files are
removed on success or exception. Original input files are never written.

The writer accepts a `DenseFlow` or floating two-channel array. **Precision**
defaults to Float32 only; **Convert float64** explicitly permits narrowing and
reports its maximum absolute rounding error. Finite values that overflow float32
are always rejected. **Invalid vectors** rejects nonfinite or Middlebury unknown
values (>1e9 in either component) by default. Preserve retains their exact float32
samples and reports their counts; it does not turn them into valid motion. The
loader preserves these values by default for compatibility, with an optional
Reject policy. Motion consumers still apply their own support/motion limits.

The writer verifies the native serialized samples before publication. It creates
files exclusively and preserves existing `.flo` files and companions. **Unique
name** adds a UUID for repeatable graph runs. **Create directories** optionally
creates output folders after input validation. Project-relative output paths
follow Save As to the new project directory. A wired loader **Path** can consume
the writer's actual output filename directly.

Both tools expose Flow and Info. Optional Visualization uses HSV direction and
magnitude, with **Visual max motion** setting full brightness; invalid vectors
are black. Diagnostic images follow the graphics setting, while an explicitly
exposed Visualization output is computed when needed. Lean/off can skip these
optional images without changing field samples. No image mask is needed for
transport: file dimensions and every vector are preserved.

## Pair manifest

`.flo` itself stores no image identities, direction, pyramid level or coordinate
transforms. An optional adjacent **`filename.flo.json`** records:

- File SHA-256, shape and declared direction: source-to-target, target-to-source
  or unspecified.
- Optional Source and Target endpoint labels, pixel/type/encoding fingerprints,
  spatial transforms and pyramid metadata. Both anchors are required for a pair.

Writer **Metadata mode** chooses None, If connected or Require images. If connected
still writes direction/hash metadata when no image anchors are supplied; this is
not a verified image pair. Source/Target images are anchors, not image outputs.
Labels are user descriptions, not verified acquisition identities. Timestamps,
live ticket/outcome data and arbitrary payloads are not serialized into the file.

Loader **Metadata mode** selects Auto, Ignore, Required or Verify images. Auto
reads a present companion and rejects stale/malformed metadata rather than falling
back silently. Required needs a valid companion. Verify images additionally needs
both anchors and compares exact pixels, sample types, encoding and coordinate/
pyramid geometry. Supplying anchors in Auto also requests that verification.
Ignore treats the file as unassociated, including when anchor ports remain wired.
**Expected direction** can require a particular declaration; an untagged file
cannot establish one.

After verification the loader binds the field's index grid to the current Source
or Target anchor according to direction. Fresh root IDs are allowed when pixels
and grid geometry match. Vectors remain in local pixels; no pyramid scaling is
performed. Stored ticket IDs are never restored; loading issues a fresh source
ticket. Normal ticket/payload propagation remains available on connections.

Warp image with dense flow and Refine dense optical flow check file-associated
fields again: changed field samples, contradictory directions, wrong image pixels
or changed geometry are invalid inputs with no result emission. Refinement expects
source-to-target flow. Warping can check Source alone when no Target anchor is
connected and reports which images were checked. Bare fields retain their existing
explicit-direction/size-only behavior. The caller must still select the intended
pair and direction for unassociated files.

Fingerprints establish byte-level image/grid identity. They do not prove camera
identity, unique acquisition time, physical calibration, visibility or estimated
motion accuracy; identical captures can share a fingerprint. Metadata is strict
finite JSON, at most 64 KiB, without duplicate keys. It is hash-bound, not signed.

Companions publish before the flow file so a newly published pair is not initially
missing its manifest. On a handled publication failure, rollback removes only
unchanged files created by that operation. There is no filesystem-wide transaction
or power-loss guarantee across the two files; an orphan companion can remain after
a crash and is preserved as a destination conflict on the next write.
The output filesystem must support hard links; unsupported locations fail without
overwriting existing files.

[Five original examples](<../examples/opencv tools/flow_file_roundtrip/README.md>)
include exact signed translation, measured Farneback motion, round trips, bare
interop and deliberate wrong-pair/direction rejection. See the
[OpenCV APIs](https://docs.opencv.org/5.0/main_modules/video_track.html) and
[Middlebury format definition](https://vision.middlebury.edu/flow/code/flow-code/README.txt).
