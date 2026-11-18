# Review a completed PathWidth run

SimplePathWidth **0.1.0.dev13** adds **Review completed path width…** in the
host inspector's Results page. This separate read-only action does not open or
modify training. It requires the host's paired execution snapshots and plugin
Results hook, plus a completed run with input/output previews retained (Debug).

## Workflow

1. Run an inspection, select the PathWidth block and open its Results review.
   The popup captures the exact consumed frame, executed manifest, effective
   fixture transform/calibration and immutable measured station results.
2. Click near a station in the image or along either trend; use the station
   number, problem filter and Previous/Next to review gaps and violations.
   Pixel and physical units stay separate. The full-path verdict includes
   stations outside the selected view.
3. Scroll to zoom, drag to pan, choose Focus station, Fit image or 100%.
   The exact consumed input has its own tab, preserving original channels/depth.
   The measurement view uses the actual grayscale inspection grid.
4. Export station… writes a new folder. Closing review leaves parameters,
   training drafts and the completed result unchanged.
5. A newer publication marks the view as frozen; Capture latest completed run
   explicitly replaces it. Active execution, unavailable retention or an
   exception refuses capture while keeping an existing frozen frame labeled.

Green denotes measured passing stations, red tolerance violations/missing pairs,
amber ambiguity, gray unobserved scans and cyan the selected station/scan/edges.
Missing pairs count toward gaps. Masked, clipped, corner and ambiguous stations
remain separate, with null widths rather than zero. Actual executed settings
provide the limits; diagnostic graphs introduce no new acceptance gates.

## Performance and identity

Execution retains only one immutable model/measurement/context record when both
previews are retained. It does not copy another source image, load the training
reference or sample extra profiles. Lean skips the additional record. Worker to
display-graph transfer shares the immutable record. Reset/close releases it.

On explicit review capture, the SDK atomically copies matching completed input
and output packets. A namespaced metadata token binds them to the cached record;
the primary numerical JSON and default graphics remain unchanged. Input tickets,
timestamps, spatial and pyramid mappings are retained. Software counters and
snapshot identifiers do not establish physical exposure identity.

The existing algorithm does not retain all sampled profiles. Selecting a station
therefore resamples only that diagnostic intensity/gradient trace from the frozen
pixels and executed calibration/settings. A 64-station cache reuses these samples.
It does not rerun edge pairing, continuity, width measurement or the verdict.
Unobserved stations do not gain invented samples or measurements. Profile values
keep native input intensity units; gradients remain intensity per image pixel.

Changing/deleting a recipe file, changing the current upstream frame or editing
an unapplied training draft cannot replace the reviewed model/frame/limits.

## Export

The existing station review images remain: overlay.png, profile.png,
pixel_trend.png, optional physical_trend.png and assessment.png. The assessment
is a display-only crop/chart; its source crop coordinates are recorded. The
review JSON includes measured values, nulls, limits and native profile samples.
Completed-run export adds:

- consumed_input.npy: exact original consumed pixels, dtype and channels.
- measurement_input.npy: exact grayscale pixels used for measurement.
- inspected_mask.png: included/excluded input pixels; station status still
  determines whether an entire scan was observable.
- executed_manifest.json: original executed manifest bytes and their pinned SHA.
- completed_execution in review.json: full-path results, station-group summary,
  effective pose/calibration, exact input hash and completed-packet provenance.

NPY files use no pickle. An existing destination is refused. The manifest is not
a full `.spw` training archive: the runtime intentionally did not load its
training reference. Current inspection pixels are never substituted for that
reference. These diagnostics do not establish optical/metrology accuracy.
