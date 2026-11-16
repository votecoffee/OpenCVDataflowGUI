# Registered-view replay

**Replay registered views** (`cv.replay_registered_views`) places each image
independently on one saved planar mosaic grid. Frames and Coverage are matching
ordered groups of images and validity masks. Connect Frames to **Unbatch serial
entries** for per-ticket inspection or recording. Image previews one selected
frame; Common identifies support shared by all received views; Info reports
geometry and source matching. Replay performs no blending or feature fitting.

Use a successful layout from **Register scan layout** or **Load mosaic layout**.
The [layout reuse contract](mosaic-layout-reuse.md) defines legacy ordered v1,
stable-ID v2, grid validation, missing policies and saved assets. Replay uses those
same rules. One to 64 received images are accepted; v1 requires the complete
original ordered group. Optional uint8 masks match received order and source
coordinates. Source IDs follow the saved rule, a nonempty explicit list or the
connected Source IDs input.

## Controls and validity

| Control | Behavior |
| --- | --- |
| Crop `canvas` | Full calibrated grid, including areas outside every received image. |
| Crop `union` | Tight bounding rectangle of actual valid received samples. Each frame retains its own support. |
| Crop `intersection` | Tight bounding rectangle of support valid in every received image; all frames are restricted to that common mask. |
| Backend `preserve` (default) | Perspective resampling with unchanged uint8/uint16/float32/float64 gray/RGB/BGR(A) depth and encoding. |
| Backend `native_detail` | Timelapser placement, with exact uint8 gray/RGB/BGR conversion through signed int16 color. Other depths/alpha are rejected. |
| Interpolation | Nearest or linear. Linear requires all contributing source samples to be valid. |
| Frame order | Received (default) keeps acquisition order; Calibrated uses saved record order. |
| Preview frame | Zero-based output entry shown as Image; other frames are retained. |
| Maximum workspace MiB | Conservative work-buffer estimate, 1–256 MiB; default 256. |

An intersection bounding rectangle is not a largest all-valid rectangle: source
masks and projective borders can leave holes. Invalid pixels are zero, and Common
identifies exact shared support on the chosen output grid. Use Common or per-frame
Coverage for measurement; zero intensity alone does not mean invalid. When views
are missing, common support is computed across **received** sources. No earlier
acquisition is substituted.

No valid samples, or no common support in Intersection mode, yields a valid failed
judgment with reason `replay_no_valid_support` or `replay_no_common_support`.
Info and a zero Common mask on the full calibrated grid are emitted; Frames,
Coverage and Image emit nothing. Native placement is not initialized. An empty
individual source in Canvas/Union mode remains a blank frame with zero coverage.
Missing policy `fail` emits failed partial data with `mosaic_sources_missing`;
`allow` can pass explicitly incomplete data. Invalid IDs/grids, unsupported native
types, preview indices and resource limits reject the invocation with no events.

## Tickets and coordinates

Every frame and matching coverage member retains its current ticket, timestamp
fields and custom payload. Replay does not synthesize capture times or renumber
tickets. Received order does not establish chronological or synchronized exposure;
use ticket coordination or Unbatch sequence sorting when needed.

All members share output dimensions and grid. V2 uses `mosaic:<layout_id>` and
the full calibrated root size even without the anchor view. Cropping translates
`spatial.to_root` by the reported origin. V1 keeps its existing anchor convention.
The `registered_replay` member payload records output/calibrated/input indices,
source ID/label, current source ticket, original spatial frame, original pyramid
descriptor, `source_to_mosaic`, `source_to_frame`, valid count, origin and layout
identity. Resampling removes the old top-level pyramid field because the result
is no longer a source pyramid level; its original descriptor remains in the record.

Info includes union/common bounds and pixel counts, matching status, backend/order,
buffer estimate and all frame records. Image carries the selected member payload.
Common describes shared geometry. Ordinary exception containment, logging, timing
and outcome stamping apply, including a failed outcome on every emitted member
when the replay fails. Lean/graphics-off retains the primary images and masks.

## Native placement and bounds

The installed OpenCV 5.0.0/contrib 5.0.0.93 Timelapser accepts signed-int16
three-channel input and ignores its mask argument. Independent local probes and
the [upstream implementation](https://github.com/opencv/opencv/blob/5.x/modules/stitching/src/timelapsers.cpp)
confirm this behavior. The wrapper resamples and zeros invalid samples before
placement, then reapplies exact validity after cropping. Uint8 conversion through
int16 is exact; output retains the original gray/RGB/BGR encoding.

Union/Canvas use `Timelapser.createDefault(AS_IS)`. Intersection uses
`Timelapser.createDefault(CROP)` and narrows its rectangular crop to exact common
support bounds. The direct Python `TimelapserCrop()` constructor raised an unknown
C++ exception with an access-violation diagnostic in this wheel; the isolated
factory audit passed, so application code uses the factory exclusively. See the
[native API reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1detail_1_1Timelapser.html).
Native objects are local to one invocation and outputs do not share its reusable
destination buffer. No performance advantage is claimed.

Dense replay allows canvas sides up to 16,384 and rejects its buffer estimate
before allocating canvas work arrays or entering native warping/placement. The
estimate includes retained frames/masks, support, warping and native temporaries.
Existing source limits remain 128 MiB total, 8,388,608 pixels per image and sides
below 32,767. Input snapshots, graph copies, retained history/previews and allocator
overhead are additional. Tight cropping does not bypass full-grid preflight.
Use **Render mosaic tile** for large canvases.

[Nine original MIT projects](<../examples/opencv tools/registered_view_replay/README.md>)
cover union/intersection, native masks, reordered/missing inputs, uint16, no common
support and explicit native-depth/workspace rejection. Unbatch sends individual
ticketed frames to statistics, with Common selecting valid pixels in intersection
examples. Focused checks include independent pixels, half-pixel averages, rotation,
depth/encoding, masks, coordinates, timestamps/pyramid context, native reset,
parallel instances, exception recovery, XML/reopen and offscreen controls. Saved
examples reproduce original graph/image/layout bytes in Debug and Lean. These
fixtures qualify planar replay, not physical camera identity, metric calibration,
simultaneous exposure or arbitrary registration accuracy.
