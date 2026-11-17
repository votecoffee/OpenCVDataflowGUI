# HDR brackets with explicit measured support

**Merge HDR with motion rejection** (`cv.merge_hdr_motion`) combines aligned,
timed uint8 grayscale, RGB or BGR exposures. It excludes pixels that disagree
after camera-response and exposure-time normalization. Connect **Image** and
**Valid mask** to measurements. Use **Display** for review: its reference fills
are explicitly separate from measured HDR.

This is a response-aware weighted linear-radiance workflow implemented with
OpenCV lookup/registration operations and bounded array accumulation. It is not
native MergeDebevec or MergeRobertson with an added mask argument. Existing
calibration, ordinary merge, Mertens fusion and tone-map blocks remain available.

## Inputs and registration

Supply an ordered 2–16-image exposure batch and durations in seconds, in the same
order. Times input overrides the Exposure times field. At least two durations
must differ. Reference index chooses the output grid and review image; it does
not make the other images simultaneous.

Use **Align exposures (MTB)** before merging if there is camera displacement.
Its output retains member labels/tickets and declares `exposure_support` metadata
(`opencv_mtb_support_v1`): original size/spatial context, integer shift, output
rectangle and resulting spatial frame. The new merge validates this grid and
automatically excludes shifted padding. Cropped overlap and zero-padded canvases
are both supported. A later resampling cannot reuse a stale support declaration.
The metadata records declared geometry, not authentication of arbitrary edited pixels.

**Grid policy** normally requires known matching pixel frames. For independently
acquired but already aligned images, **Assert pixel alignment** records the
author's explicit same-index assertion. Equal dimensions alone cannot verify
registration or camera identity. The examples use this assertion only for
authored aligned controls; the shifted control runs native MTB.

Optional **Masks** is a uint8 mask batch with the same labels/order as Exposures.
Each mask must match its corresponding image dimensions and known frame. It
intersects automatic alignment support. Masked pixels supply neither radiance nor
clipping evidence. Reference **Mask** and **Region** intersect to define the
measurement domain. Standard shapes, holes, compositions and raster regions work.

## Radiance and consistency

Without Response, the assumed linear curve is `g(z)=z/128`. An existing
Debevec/Robertson response output can replace it. Response channel order is
interpreted against the batch encoding. For each usable exposure, radiance is
`g(z)/seconds`. These are response units per second, not absolute radiometry.
Camera response, actual exposure duration, lighting, noise and registration must
be appropriate for the inspection. Values are not independently exposure-metered.

An exposure contributes only when every stored channel is strictly between
**Clip low** and **Clip high**, its response radiance is positive and its masks
allow the sample. Defaults accept raw codes 5–250. **Weighting** is uniform, or
the triangular distance from the nearest clipping boundary, using the least
well-exposed channel for a shared color weight. Accepted radiances are averaged
in linear units. Float64 accumulation is converted to finite float32 output.

Consistency is checked per channel using the full observed radiance span. Allowed
disagreement is **Absolute tolerance** plus **Relative tolerance** times the larger
observed bound magnitude. All participating exposures must be compatible: the tool
does not choose a majority background over a moving reference object. A disagreement
in any channel rejects the entire pixel from measured HDR.

**Clipping evidence** defaults to **Bounds**. A high-clipped code gives a lower
response bound; a low-clipped code gives an upper bound. Extrema over the excluded
response-code bands make these bounds conservative for the supplied lookup table.
These constraints can contradict the unclipped observations even though the clipped
exposure is not usable for radiance. This catches the example's moving part in the
long exposure. **Ignore** omits clipped observations entirely and deliberately has
less evidence. All-clipped pixels remain unavailable; bounds do not reconstruct
their radiance. Physical interpretation assumes the clipping/response model fits
the camera. Blooming, nonmonotonic response, illumination changes and registration
errors can also produce disagreement.

At least **Minimum contributors** (default two) unclipped agreeing observations
are required for a measured pixel. One reference exposure cannot become measured
HDR merely because it is available for display. **Minimum coverage** applies to
the selected reference domain. Empty or wholly unsupported domains always fail,
even with minimum coverage zero. A valid coverage shortage is an ordinary failed
inspection with diagnostic outputs retained; malformed inputs, settings, stale
support or native errors are contained block exceptions without current events.

## Outputs and provenance

| Output | Contract |
| --- | --- |
| Image | Float32 measured radiance, zero everywhere outside Valid mask. |
| Display | Review radiance: measured pixels plus optional reference fills. |
| Valid mask | Binary uint8 measured support; use this on downstream measurements. |
| Motion mask | Photometric inconsistency, including configured clipping contradictions. |
| Fallback mask | Display pixels supplied by the reference rather than measured HDR. |
| Contributors | Accepted exposure count, zero outside measured support. |
| Support masks | Optional per-exposure accepted masks in the original member label/order. |
| Report | Coverage, clipping/zero-response/mask rejection, policies, time and source provenance. |

**Fallback** chooses reference radiance or zero for Display gaps. Reference fills
may themselves be clipped and remain review-only; Fallback mask marks them even
outside the selected measurement region. Display must not be treated as measured
HDR. For a moving part, inspect its original reference exposure in a separate
branch if a single capture is the intended measurement.

Outputs keep the aggregate bracket ticket, reference spatial grid and member
ticket/spatial/alignment records. Member support masks retain member provenance.
`hdr_support.requires_valid_mask` marks the measurement contract; Display also
has `review_only: true`. These tags do not automatically mask arbitrary downstream
blocks. Explicitly connect Valid mask. The saved examples do this for Image statistics.

Drawable red outlines identify inconsistent regions on Display; coverage text and
the filmstrip preserve current diagnostics on a failed inspection. Drawings use a
262,144-pixel preview and the configured point budget. They never alter numerical
outputs. Lean or graphics-off skips them; a zero point budget emits no drawings.
The optional accepted mask batch is disabled by default to reduce allocations.

## Bounds and examples

The default reference grid limit is 4,194,304 pixels, with axes at most 16,384.
The inherited exposure stack also caps 24 million stored channel samples and
128 MiB of source images. Maximum working bytes defaults to a 768 MiB array
estimate, checked before radiance allocation. Accumulation streams one radiance
frame at a time instead of retaining a full float64 radiance stack. Runtime
copies, Python metadata and native workspaces are outside the estimate; it is not
a process RSS guarantee. The tool is stateless, preview-safe, parallel-safe and
readonly. XML/editor controls use the common contracts.

Try [six original projects](<../examples/opencv tools/hdr_motion/README.md>) for
static radiance, moving/clipped parts, weaker clipping evidence, all-clipped
missing support, strict coverage failure and native shifted-frame alignment.
Independent source scenes specify exact values, masks, counts and statistics.
Photometric disagreement is a motion proxy, not foreground segmentation, recovered
hidden area, physical timing synchronization or absolute calibration.

Native references: [OpenCV HDR imaging](https://docs.opencv.org/4.13.0/d6/df5/group__photo__hdr.html),
[MTB alignment](https://docs.opencv.org/4.13.0/d7/db6/classcv_1_1AlignMTB.html).
