# Motion-compensated frame interpolation

Open `project.json` and run once. The standard application installation is
sufficient. The original 16-bit color fixture moves eight pixels right and four
pixels down between its endpoint frames. An independent crop of the larger
original texture supplies the expected halfway image.

The graph estimates both flow directions, creates the halfway image at 20 ms
between configured 0 ms and 40 ms endpoints, and measures its error on supported
interior pixels. On the pinned OpenCV 5 build:

| Measurement | Result |
| --- | ---: |
| Whole-frame motion support | 94.8% |
| Selected interior support | 100% |
| Interpolated interior mean absolute error | 19.9 on the uint16 scale |
| Plain endpoint crossfade interior mean absolute error | 7,122 on the same scale |
| Interpolated timestamp | 20 ms |

The saved quality check allows a maximum mean error of 100. Results can vary
across builds. This fixture checks small translation with texture; it does not
qualify arbitrary real video, complex occlusion or capture synchronization.

From the repository root:

```powershell
opencv-dataflow-run "examples/opencv tools/frame_interpolation/project.json"
opencv-dataflow-run "examples/opencv tools/frame_interpolation/project.json" --profile lean
```

Regenerate the image assets and graph with:

```powershell
python "examples/opencv tools/frame_interpolation/generate.py"
```

## Motion and support

**Motion-compensated frame interpolation** operates on one explicit pair of
frames. Set `fraction` between zero and one. Its flow modes are:

- **Estimate:** compute forward and backward Farneback flow.
- **Refine:** use wired fields as Farneback initial estimates.
- **Provided:** use wired fields directly, without estimating flow.

Forward vectors describe A-to-B displacement at A's pixels; backward vectors
describe B-to-A displacement at B's pixels. Inputs accept the existing DenseFlow
type or finite float32/64 H×W×2 arrays. Connect support masks when fields include
unknown or untracked areas. All Farneback pyramid, window, polynomial, iteration
and Gaussian weighting controls are exposed. Estimation uses an 8-bit luminance
copy; the output retains its original encoding and pixel depth. For HDR data
outside [0,1], provide flow calculated from a suitable detection image.

The block solves each inverse motion map iteratively, checks convergence and
source bounds, and checks forward/backward closure. Optional appearance tolerance
rejects corresponding luminance changes on an 8-bit normalized copy. Zero
disables that test. Illumination changes can also trigger it. These checks are
support/occlusion proxies; consistent wrong motion can pass them.

Supported source samples blend with normalized temporal weights. A single
supported sample is used when its counterpart is unavailable. `uncovered`
chooses zero, the nearer unwarped endpoint, or an unwarped crossfade for holes.
Fallback pixels remain uncovered in every case. `minimum_coverage` can fail the
block with a reusable failure reason when support is insufficient.

Image sampling offers nearest, linear, cubic and Lanczos4 with accurate native
remapping. Cubic/Lanczos support conservatively excludes a two/four-pixel source
rim. Endpoints at fraction zero/one copy selected pixels exactly without flow.
Images are limited to 2.1 MP and fewer than 32,767 pixels on either axis. Motion,
inverse iterations and tolerances are bounded and validated.

See the [OpenCV Farneback API](https://docs.opencv.org/5.0/main_modules/video_track.html#calcopticalflowfarneback)
and [remap API](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html#remap)
for the underlying native operations. Their installed binding reference is also
available in the block's documentation view.

## Outputs, selection and coordinates

| Output | Meaning |
| --- | --- |
| `image` | Generated image with the source encoding and depth. |
| `coverage` | Uint8 target-grid support mask: 255 for a supported selected pixel. |
| `info` | Coverage counts, fraction, source identities, thresholds and timing provenance. |
| `timestamp_ms` | Interpolated media/explicit time; no event when timing is unavailable. |
| `coverage_a`, `coverage_b` | Target-grid support from the respective source. |
| `uncertain` | Selected intermediate pixels lacking support from both sources. Exact endpoints report zero uncertainty. |
| `forward_flow`, `backward_flow` | Optional DenseFlow outputs, enabled with `emit_flows`. Each retains its source pixel grid. |
| `coverage_preview` | Both sources green, A only amber, B only cyan, neither red; unselected black. |
| `overlay` | Interpolated image with a drawable coverage label. |

The first four outputs are exposed by default; enable additional connectors in
the inspector. Graphics disabled skips previews and drawable graphics. Numeric
images and coverage remain available. Flow retention is a separate option.

Frames must have matching shape, depth and grayscale/RGB(A) encoding. Same-frame
alignment checks coordinate identity. **Pixel grid** explicitly pairs images by
index across identities, as used by the independent file sources in this graph.
It assumes the grids are already registered. HSV/Lab and physical DepthArray
inputs are rejected. Alpha channels use ordinary component interpolation.

Masks and circle/rectangle/polygon regions intersect to select output centers
on A's grid. Motion is evaluated using complete frames. Outside the selection,
pixels remain from A. Tickets, frame/pyramid and companion payloads from A remain
on the output wires; both sources' ticket, frame, recipe and media identities are
retained in `frame_interpolation` lineage. No new acquisition ticket is invented.
The backward field uses B's grid while retaining the output job ticket from A.

## Timing

Media timing is inferred from `video_frame` only when both frames share a known
source-grid identity and video filename. Explicit timing uses configurable A/B
times, or a pair of wired timestamp inputs that overrides configured timing.
Times must be finite, nonnegative, strictly ordered and in the same clock domain.
The required policy rejects missing compatible times; the default reports
unavailable timing explicitly.

For a compatible video stream, the output has an interpolated timestamp and a
synthetic flag, with the original integer frame index removed. Unrelated streams
do not inherit A's acquisition timestamp as the output time. Original media
contexts remain in lineage. These are media times, separate from the runtime's
ticket processing start/finish times.

This pair block emits one image per activation. Automatic video-rate expansion
and paced playback are separate sequencing concerns.

## Attribution

All images, generator code and the project are original examples under the
repository's MIT license. No customer images or downloaded assets are included.
