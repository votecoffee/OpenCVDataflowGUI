# On-demand TIFF pages

`input.tiff_page` (**OpenCV → Image Codecs (imgcodecs) → Files → Read TIFF page
on demand**) reads one zero-based page whenever the graph runs. A wired **Page
index** overrides the configured value and must emit before the connected read
can run. Requests can go backwards, repeat or jump to any inspected page. For
automatic sequential replay use `input.tiff_file`; for encoded image groups and
application sidecars use [multipage TIFF buffers](multipage-memory.md).

The supported Python wheel does not expose `ImageCollection`. This block uses
native `imdecodemulti` with a one-page range and a Python LRU of raw decoded
samples. Intermediate TIFF directories still need traversal on a miss; there is
no constant-time random-decode claim. See the [five saved examples](<../examples/opencv tools/tiff_page_access/README.md>).

## Snapshot and supported pages

The source reads a bounded immutable file snapshot and inspects its directories
before pixel decoding. It supports the shared directory reader's top-left,
contiguous gray/RGB(A) classic/BigTIFF subset: uint8, uint16, float32 and float64,
with uncompressed, LZW, Deflate or PackBits storage, up to 1024 directories and
128 MiB encoded bytes. Other TIFF layouts/compressions are rejected. Directory
inspection counts pages without decoding their pixels. No source file handle or
opaque native decoder is retained between calls.

Output **Image** is a private converted page. Unchanged preserves native
gray/BGR/BGRA order and sample depth, including finite negative/HDR float values.
TIFF RGB storage is decoded in BGR order. Color/depth controls use the standard
conversion rules; unsupported HSV/Lab depth combinations are rejected before
native pixel decoding. Conversion does not alter a cached raw page.
**Page count** and **Info** expose the inspected count, selected declaration,
container SHA-256, output format and cache observations.

## Cache and refresh policies

**Cache pages** and **Cache bytes** jointly limit raw decoded pages using
least-recently-used eviction. A zero limit disables retention. A single page
larger than the cache byte budget is decoded for the current output and bypasses
retention. Limits are reapplied on each activation, including after tightening
them. Hits avoid native pixel decoding, but still make a private formatted output.
The encoded snapshot, directory metadata and output/history copies are separate
from the raw-page cache budget.

- **on_change** checks resolved path, file identity, size and modification time
  (also change time on POSIX). Stable observations reuse the snapshot. A changed
  observation reads/hashes a new snapshot; identical content keeps its cache.
- **verify_content** reads/hashes on every request, detecting changes even when
  file observations are unchanged. It costs file I/O and hashing on cache hits.
- **pinned** reuses the snapshot without rechecking the file until reset or a
  configured/resolved path change. It can deliberately keep serving the old
  snapshot after the disk file changes or disappears.

File size is checked before allocation, with size/time checks during reading.
These checks do not make file observations proof against all same-size/time
rewrites; Verify content provides content comparison. Reported hashes identify
the bytes actually decoded. A snapshot change invalidates cached pixels.
Missing/invalid files release the previous snapshot instead of silently using it.

**Reset=true** discards snapshot, cache and counters before that request. Runtime
reset/close does the same. Applying block configuration can also reset runtime
state through the editor's shared lifecycle. For interactive page selection
without changing the TIFF block, wire a selector such as Counter input with
Step=0 and edit its Start value, as in the example.

## Tickets, coordinates and lifecycle

Automatic tickets use `filename#page` with a fresh acquisition identity for every
request, including cache hits. External/fixed/timestamp ticket settings use the
shared source rules. A pixel grid is bound to block ID, container content hash
and page index, so revisiting a page preserves its grid while changed contents
produce a different one. This is pixel provenance, not a camera identity or a
physical calibration. TIFF files do not restore application payload sidecars;
use the typed buffer tools for that workflow.

Raw arrays/snapshot state copy into independent editor execution instances.
Closing a worker cannot clear the owner's cache. JSON/XML save controls and
layout, not cached pixels; reopening starts empty. Source blocks use serial
execution for their mutable cache. Preview is eligible and still follows normal
validation, ticket, failure-policy, logging and timing behavior.

Maximum page pixels and a conservative raw/conversion workspace estimate are
checked from directory declarations before pixel decoding. Retained cache,
encoded data, runtime copies and native codec internals are outside the workspace
estimate; these are not hard RSS limits. Cancellation is cooperative and cannot
interrupt a native decode already in progress. Invalid requests or native
failures emit no partial image or info event; an unaffected cached page can still
be used by a later valid request.

## Evidence

A 257-page original MIT stack alternates uint16 gray, uint8 BGR, float32 gray and
float64 gray with known pixel hashes and negative/HDR samples. In the fixed
11-request order, a two-page cache makes nine native one-page calls and two hits;
peak raw cache residency is 196,608 bytes versus 13,393,920 raw bytes for all
pages. These are native-call/residency observations, not an application speedup
claim. Tests verify exact pixels, eviction and bypass, private outputs, refresh,
reset, Unicode paths, external tickets, worker copies, Save As/XML, offscreen
dropdowns and controlled invalid/cancel/native-error cases.

Upstream: [indexed buffer decoding](https://docs.opencv.org/5.0/main_modules/imgcodecs.html#imdecodemulti),
[ImageCollection](https://docs.opencv.org/5.0/main_modules/classcv_1_1ImageCollection.html).
