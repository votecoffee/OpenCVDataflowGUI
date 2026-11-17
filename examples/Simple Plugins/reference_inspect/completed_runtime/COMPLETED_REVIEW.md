# Review a completed reference inspection

SimpleReferenceInspect **0.1.0.dev4** adds **Review completed reference inspection…**
to the inspector's Results page, separate from training. Use a host with paired
completed snapshots and the read-only plugin Results hook. Run in Debug with
input/output previews retained; changing profile alone cannot recreate old data.

## Workflow

1. Run, select the inspection block and open its Results review. It captures exact
   consumed pixels, the executed trained model and cached reference/source maps.
2. Compare the reference mean beside normalized current, signed residual,
   deviation/tolerance or inspected coverage. Both views share the reference grid
   and intensity scale. Scroll zooms, drag pans both; use 100% or Fit comparisons.
3. Input overlay retains consumed-input coordinates. Assessment is a separate
   display grid. The outer Exact consumed input tab retains original depth/channels.
4. Review actual limits/gates, support counts and failed reasons. Prerequisite
   failures stay Not evaluated; accepted area/max deviation are unavailable, not
   zero measurements. A tolerance ratio is not a class probability.
5. Export review… writes a new immutable-frame folder; close leaves parameters,
   training drafts and the completed result unchanged.
6. New editor publications mark the view frozen. Capture latest completed run
   explicitly replaces it. Active/error/unretained input refuses capture and keeps
   a previous frozen view clearly labeled.

Coverage colors: green inspected, red accepted change, amber expected but not
inspected, dark excluded. Gray residual/deviation pixels were not inspected.
Signed residual is native normalized-current minus reference intensity. Deviation
is absolute residual divided by the trained tolerance; one is the pixel threshold.
Connected-component area filtering and full inspection gates still own the verdict.
Geometric support, runtime selection and inspected-pixel counts stay distinct.

## Execution and identity

The Debug record retains the already computed private reference-grid buffers,
executed model, small report and transform. Published source maps are separate.
No additional raster/model copies, resampling, exposure fit or evaluation are
added to execution. Lean skips this record. The runtime still verifies its recipe
pin using its existing loader; review itself never reloads that file.

The SDK atomically copies matching completed input/output packets on explicit
capture. A namespaced output-metadata token binds that completion to the private
model/reference maps. Numerical JSON and existing output maps/graphics are unchanged.
The captured reference/source buffers and model are private immutable byte-backed
arrays. A frozen model cache is reused across captures of the same loaded model;
model changes, Lean execution, reset and close release it.

The comparison's normalized-current view resamples frozen measurement pixels and
applies recorded pose/gain/offset. It does not refit exposure, compute residuals,
segment components or decide the verdict. Source overlays reuse captured output
maps. Input tickets, timestamps, spatial and pyramid metadata remain explicit;
software completion/snapshot IDs do not establish physical exposure identity.

## Export

The existing seven comparison/assessment PNGs, review JSON and four native
reference-grid result arrays remain. Completed-run export adds:

- consumed_input.npy and measurement_input.npy: exact original consumed array
  and native grayscale pixels used by inspection, with no pickle.
- source_residual.npy, source_deviation_map.npy, source_inspected_mask.npy and
  source_defect_mask.npy: exact completed outputs in the input grid.
- normalized_current.npy and normalized_support.npy when exposure coefficients
  are available: reconstructed diagnostic current/support in the reference grid.
- executed_model.sri: re-encoded executed model with all aligned training samples,
  support, masks, statistics, settings and captured reference fixture.
- completed_execution in review.json: original pin/exported archive hash, map
  dimensions/dtypes/hashes, exact input hash, pose/exposure and completed provenance.

Re-encoded archive bytes may differ from the original pin. Export refuses an
existing folder and never changes the active recipe. Source/reference/display
grids and reference-pixel component areas stay explicit. These reviews do not
establish independent accuracy, defect labels or operator acceptance.
