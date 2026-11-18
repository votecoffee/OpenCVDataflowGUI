# Review a completed segmentation

SimpleSegmentation **0.1.0.dev2** adds **Review completed segmentation…** in the
host inspector's Results page, separate from annotation/training. It requires
paired completed snapshots and the independent read-only Results hook. Run in
Debug with input/output previews retained; capture-only mode does not segment.

## Workflow

1. Run the whole-image or matched-parts block and open its Results review. Exact
   consumed pixels, original pinned model/region bytes, executed settings and
   completed source masks/coverage are captured together.
2. Choose a reported part/image. Its table lists connected components in that
   part's actual sampled grid: bounds, area and local mean forest vote.
3. Select a table row or click its projected bounds in a source view. Cyan marks
   those bounds and center. Focus selected zooms to it; All parts overview shows
   the full source union. Drag to pan and scroll to zoom.
4. Switch between source overlay, votes, coverage, exact original input and an
   assessment page. The native input tab preserves original depth/channels.
5. Export frozen review… writes a new folder. Closing review leaves configuration,
   annotation labels, training candidates and the completed outputs unchanged.
6. New publications mark the view frozen. Capture latest completed run explicitly
   replaces it. Active/error/unretained capture refuses to mix frames, keeping a
   prior frozen review clearly labeled.

Scores are retained only when Emit score map or Render heatmap was enabled for
that run. If both were disabled, no votes are fabricated or inferred during review.
Enable one option and run again to retain votes. Executed cutoff/minimum area
remain read-only: this dialog does not rethreshold or reselect components.

## Coordinate and score meaning

The overlay is the completed source mask union. Green is inspected background,
red foreground, gold coverage boundaries and gray checks excluded/unobserved.
Votes use a fixed 0..1 Viridis scale, not auto contrast or calibrated probability.
Uninspected source scores are zero sentinels; the inspected mask determines
availability. Source overlap uses maximum votes and nearest-neighbor projection.

Reported component bounds, area and mean score are measured before projection,
on each sampled patch. Sampling factors and part poses change their relationship
to source pixels. Components from separate/overlapping patches can merge in the
displayed union. A selected projected rectangle is a footprint, not an isolated
component pixel mask. The dialog does not invent missing local member masks.

The source union counts and sampled part/component counts stay separate in the
summary and export. No placements, no observed pixels and observed empty
foreground are distinct states. Empty observations have a null foreground
fraction. The block passes when some observations were processed; foreground
presence is not itself a quality pass/fail criterion.

## Execution and provenance

The existing verified-asset cache retains the original immutable YAML/PNG bytes
alongside the decoded model. A Debug-only record binds those assets, settings and
reported measurements to a completion token in output metadata. It retains an
already computed private score map only for heatmap-only runs; emitted scores
come from the paired SDK output snapshot. No additional image copy, scoring,
projection, component selection or model serialization is added to execution.
Lean/capture-only skip the review record. Reset/close release review state;
worker/display graph transfer preserves the record. Asset loading/hash verification
retains its existing execution behavior; review itself does no model IO/decoding.

Explicit capture freezes source/maps into private immutable byte-backed arrays.
The original executed assets can be exported even after files or an unapplied
training candidate change. Input tickets, timestamps, spatial/pyramid metadata
and match metadata remain explicit. Software snapshot/counter identities do not
establish physical exposure identity or independently validate the supplied pose.

## Export and follow-ups

The block now declares model_path and region_path as sealed training assets.
Save As copies their exact pinned bytes into the new project. Ordinary input
image paths retain the host's usual rebasing behavior. Training/validation
manifests and their annotation files remain separate source material; keep those
folders when retraining or move them deliberately with their relative references.

Exports contain consumed_input.npy, measurement_input.npy, mask.npy, inspected.npy,
optional score_map.npy, exact executed_model.yml/optional executed_region.png,
overlay.png, coverage.png, optional votes.png, assessment.png and review.json.
NPY uses no pickle. The JSON retains executed settings, primary report, selected
sampled component/projected bounds, original asset hashes, source crop and packet
provenance. Assessment is a display-only resized source crop, not another spatial
measurement grid. Existing destination folders are refused.

Exact per-component member masks/local score patches, runtime what-if settings
across overlaps and cross-frame batch review remain follow-ups. They need explicit
retention contracts; projected source scores cannot reproduce every patch-local
area filter. Independent exposure/part-family accuracy and hands-on operator
acceptance are separate from the completed-review workflow.
