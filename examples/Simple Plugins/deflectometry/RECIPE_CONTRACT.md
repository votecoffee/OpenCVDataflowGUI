# Retained training and execution contract

Sealed `.sdef` recipes retain normalized X/Y observations, their exact IDs/shift
order, source grid/setup/radiometry, source mask, phase-fit settings, comparable
coverage and optional phase-change limit, plus bounded JSON provenance. Reference
arrays and metadata are privately retained. Import normalizes/linearizes exactly
once; no source file is needed to reopen training.

The archive contains checksummed numeric `x_images.npy`, `y_images.npy`, `mask.npy`
and versioned `recipe.json`. No pickle or external members are accepted. Recipe
loading validates declared shape/count before reading arrays, NPY headers, hashes
and the combined 16-million-sample scope; expansion/archive are bounded to 160 MiB.
Saving exclusively creates a new asset, preserving old training. Host Save As
copies sealed training bytes. Ordinary runtime manifest/image paths remain file
references; keep their folders intact when moving projects.

The native editor retains reference source frames in a selectable filmstrip and
restores the last selected frame, exact mask, sequence and settings. New teaching
can select X/Y files, review/reorder observations, edit phase shifts and declare
encoding/exposure/scale without authoring JSON. Integer full-scale or normalized
floating input is the default scale inference; explicit scale overrides it.
Teach pose/exposure/order declarations are not hardware synchronization evidence.

Changing the region/sequence/radiometry/fit/decision settings clears stale views
and requires a fresh reference preview. Exact masks remain intact until a
rectangle/full region replaces them. Reference health stays separate from test
previews. Independent captures never replace training. Frozen jobs, cancellation,
draft revision and connected-publication checks prevent stale results publishing.
Stage saves a new asset and changes only the inspector draft; Apply commits it;
Cancel preserves applied parameters. Unused staged assets may remain on disk.

Runtime `image.simple_deflectometry` explicitly chooses **connected capture** or
**capture manifest**. It never substitutes reference images or ignores a wired
source. The connected array-preserving `any` port accepts:

```python
{
    "schema": "simple_deflectometry_stack/v1",
    "sequence": sequence_record,  # canonical dataclass record: x/y axes, setup/grid,
                                 # exposure_s and response; exact observed sequence
    "capture_id": "actual-capture-0001",
    "x_images": normalized_x,    # float32/64 N_x x H x W
    "y_images": normalized_y,    # float32/64 N_y x H x W
    "mask": optional_mask,       # H x W integer/bool; cannot expand taught ROI
    "metadata": {},              # bounded JSON provenance
}
```

`capture_bundle(capture, metadata)` builds this record. Runtime checks the actual
sequence, setup/grid, shifts/order, response and exposure against training. Linear
and explicit uncalibrated-code modes cannot silently match. Null exposure is
permitted only in code diagnostics. Test masks reduce comparable coverage against
the original taught ROI denominator.

Unchanged sealed recipes reuse their fitted reference and phase bases. Matching
connected wire metadata also reuses the taught sequence without recomputing its
phase bases. Each new capture still fits its own signals and validity. Independent
block instances can run on parallel graph branches. No measured throughput gain
is claimed yet; tiles/streams and further generic acceleration remain follow-ups.

Outputs: raw wrapped-change magnitude `image` in radians, all 18 numeric `maps`,
`report`, comparable `coverage`/`count`, optional `changed_pixels`, and optional
RGB `overlay`/`overview`. Sixteen named result graphics expose phase, modulation,
fit, signed changes and unavailability. Graphics-off preserves maps/reports and
judgments exactly, skipping optional rasters. A valid failed inspection retains
evidence; invalid input clears stale results through the host executor.

Whole-period phase ambiguity remains. A configured phase-vector limit is a user
inspection choice, not calibrated slope/height or statistical confidence. The
separate assumed-plane reflection API requires independently decoded screen U/V
and known camera/screen/plane geometry. This trainer never creates XYZ from phase.
