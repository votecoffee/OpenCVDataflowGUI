# Reusing calibrated mosaic layouts

**Register scan layout** (`cv.register_scan_layout`) fits overlapping planar views
once. **Render mosaic tile** (`cv.render_mosaic_tile`) applies that geometry to new
images without fitting features again. Stable source IDs let a received group
arrive in a different order or omit a camera while retaining the full calibrated
canvas. This is planar pixel registration; it does not establish metric camera
calibration, physical camera identity or exposure synchronization.

## Training and source identity

Connect a group of two to 64 overlapping images, with optional ordered source
masks, to Register scan layout. Registration uses the existing motion-model,
matching, inlier and loop-closure controls. Failed registration produces its
ordinary failed diagnostic layout, which cannot be rendered or saved as a valid
calibration. Choose a source identity rule before successful registration:

| Source identity | Where each stable ID comes from |
| --- | --- |
| `ordered` (default) | Legacy version-1 layout: exact input count, order and group labels. |
| `labels` | Each group entry's label; use stable camera names, rather than acquisition filenames. |
| `payload` | A top-level per-member payload key, selected by Source ID field; default `source_id`. |
| `explicit` | Source IDs parameter, or the optional Source IDs input, in received image order. |

The three stable modes produce `tiled_scan_mosaic_v2`. Every ID must be a unique,
nonempty string of at most 256 characters without control characters. IDs are
case sensitive and are not trimmed or normalized. Explicit connected IDs override
the configured list; a connected empty list is invalid. Ordered registration
rejects supplied IDs instead of silently ignoring them. The payload-field name
is a visible string of at most 128 characters.

At reuse, the saved identity rule selects the current images. A nonempty renderer
Source IDs parameter or connected Source IDs input overrides that rule for the
current group. It does not rewrite the calibration. Never reuse acquisition
tickets as permanent camera IDs unless the values actually identify the source.

## Reordering, missing views and selections

Render mosaic tile matches received IDs to saved records and restores calibrated
source order before blending. `prefer_first` and `prefer_last` therefore retain
the same priority when inputs arrive in another order. Unknown or duplicate IDs
are invalid. At least one received image is required.

| Missing views | Behavior when a calibrated source is absent |
| --- | --- |
| `error` (default) | Validation error before native warping; no output events. |
| `fail` | Emit the available partial tile and coverage, with a failed judgment and reason `mosaic_sources_missing`. |
| `allow` | Emit the partial tile with a passing judgment and explicit incomplete status. |

Allowing missing views does not require any particular coverage. A valid tile
outside all surviving sources is uncovered and contains zero samples. Use Coverage
or `covered_pixels` for inspection-specific minimum-support decisions. Saved past
images are never substituted for missing acquisitions.

Masks are optional uint8 validity images, one per **received** image, in the same
order and source coordinates. The renderer reorders images, masks and metadata
together. It does not independently match mask IDs. Linear interpolation requires
all contributing source samples to be valid. Coverage is 255 where any source
contributes; Overlap is 255 where at least two sources contribute. Uncovered
samples remain zero. See [multiband mosaics](multiband-mosaics.md) for blending,
phase-aligned halos, depth/alpha semantics and workspace bounds.

## Grid and coordinate contracts

Every present image must retain its calibrated dimensions, sample depth and
channel encoding. Reuse supports uint8, uint16, float32 and float64 grayscale or
RGB/BGR(A), subject to the selected blender's restrictions. Received spatial
metadata and mask grids are validated before binding.

With Grid policy `strict` (default), a calibration containing source spatial
metadata requires matching current image size, root size and image-to-root matrix
(absolute tolerance 1e-9). A known saved root ID must also match. Missing current
metadata or an unreported one-pixel crop-context change is rejected. A calibration
without spatial metadata can check dimensions only; the status explicitly says
`dimensions_only`.

`pixel_indices` is an explicit caller assertion that current samples occupy the
calibrated pixel grid. It bypasses comparison with the saved spatial frame and
reports `caller_asserted_pixel_indices`. It retains dimension/depth/encoding and
current-metadata validity checks. Neither policy detects a physically moved camera,
changed lens, dishonest IDs or incorrect metadata. Retrain when geometry changes.

Version-2 mosaics have their own root ID, `mosaic:<layout_id>`, and the original
full canvas size. Removing the first/anchor camera does not move this root or
shrink the canvas. Each tile's `spatial.to_root` translates local pixels by its
reported Origin. All four outputs share that tile spatial payload. Tile records
include:

- Calibrated `index`, current `input_index`, stable `source_id`, current `label`
  and `registered_label`.
- `source_to_mosaic` and `source_to_tile` for each contributor touching the
  requested tile; contributors used only for a blend halo are omitted.
- Current per-member `source_ticket` and `source_spatial`, rather than training
  tickets. Calibration records retain training metadata separately.
- `source_matching`: registered/present/missing IDs, completeness, missing policy
  and per-source grid-check evidence. This lists present sources even when they
  do not intersect the requested tile.

The incoming group retains its ordinary ticket semantics. Stable source matching
does not prove that member acquisitions occurred together; use control-flow
coordination when exposure/ticket association matters. Result provenance supports
back-mapping an inspection to each contributing image; blends do not describe
one unique source camera at overlap pixels.

## Saving and loading

Connect a successful layout to **Save mosaic layout** (`output.mosaic_layout`).
The Path ends in `.json`; relative paths use the saved project's directory.
`new` preserves existing files and is the default. `replace` writes a temporary
file beside the destination and atomically replaces it. Directories must already
exist. This output block is excluded from live previews. Its Info includes the
file-byte SHA256, byte count and version-2 layout identity.

**Load mosaic layout** (`cv.load_mosaic_layout`) reads strict UTF-8 JSON, rejects
duplicate keys and nonfinite numbers, and accepts at most 1 MiB. Optional Expected
SHA256 pins the exact file bytes. Save As copies this declared project asset.
Load and Save validate legacy v1 and stable v2 geometry; v1 reuse still requires
the original complete ordered group. Source images and native detector objects
are not stored in these files.

The v2 `layout_id` is SHA256 of canonical identity fields: model, canvas size,
identity rule, calibrated indices/IDs, source dimensions/depth/encoding,
transforms and source spatial frames. Display labels, acquisition tickets and
annotations do not change it. Editing geometric fields requires a new sealed
revision; a stale digest is rejected. This consistency digest is not an
authentication signature. The optional file SHA256 also pins annotations and
other exact bytes.

Full geometry, including absent views, is checked using dimensions and transforms
without allocating absent images or a dense mosaic. Sources remain bounded by
64 images, 8,388,608 pixels per image, sides below 32,767 and 128 MiB total. Canvas
sides are at most 1,000,000; rendering keeps the existing 16–2048 tile-size range
and 256 MiB work-buffer preflight. Graph snapshots and previews also use memory.
No native `Stitcher.composePanorama` execution or performance benefit is claimed;
this workflow explicitly reuses persisted planar transforms.

## Examples and verification

The [nine original projects](<../examples/opencv tools/mosaic_layout_reuse/README.md>)
include full/reordered capture sets, missing middle/anchor/single views, deliberate
missing/grid rejection, masked multiband tiles and a separate native registration
workflow. Three original MIT images have independently known integer placements.
Focused checks cover exact pixels/support/priority, calibrated order, current
tickets, stable canvas coordinates, feather/multiband adjacent-tile seams, native
fitting versus reuse, large virtual extents, strict file persistence, exception
recovery, XML/Save As and offscreen inspector dropdowns.

For separate acquisitions, [Registered-view replay](registered-view-replay.md)
uses these same layouts and source-ID rules. Frame/coverage groups share full,
union or common-valid grids, retaining member tickets and timestamps.
