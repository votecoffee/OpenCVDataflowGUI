# Acquisition identity in locator results

The `3d.simple_locate` host block in version `0.1.0.dev20` accepts an optional
**capture_id** text input and reads `capture_id` metadata on its **scene** and
optional **depth_scene** point clouds. It checks the declarations before loading
training, building the model index or running matching.

| Inputs | Behavior |
| --- | --- |
| No input or cloud IDs | Existing unbound legacy behavior; no report ID is invented. |
| Explicit ID, both clouds untagged | Bound as `explicit_input_declaration`; the caller owns their pairing. |
| Every connected cloud tagged with one equal ID | Bound as `matching_cloud_metadata`. |
| Explicit ID plus equal cloud IDs | Bound as `explicit_input_and_matching_cloud_metadata`. |
| Different IDs, invalid ID, or only one cloud of a connected pair tagged | Reject before loading or matching; no current pose is published. |

IDs are nonempty strings of at most 256 characters, without control characters.
A bound invocation emits the ID on the optional **capture_id** output, in the
report and its match records, in every accepted `pose_3d_v1`, and in native fit/
aligned-model cloud metadata. The `capture_binding` object identifies the method
and cloud inputs checked. A no-match result retains the current ID and an empty
pose list; it never reuses a pose from another acquisition.

Metadata is a caller declaration. Equal IDs do not prove exposure synchronization,
timestamp accuracy, robot pose association or camera calibration. Do not tag old
retained depth with a new acquisition ID. A tagged cloud cannot silently pair
with an untagged full-depth cloud, even when the explicit input is connected.

The numerical matcher, thresholds, settings, recipes and worker counts are
unchanged. Untagged output geometry/reports retain their original values; the
new optional output is `NO_VALUE`. A cached model index can serve successive
captures because the index depends on the trained model, not capture identity.

This contract belongs to the host's typed point-cloud adapter. The standalone
array-based `Locator.locate` API and CLI cannot infer acquisition identity from
XYZ arrays. Their caller remains responsible for pairing sources. Trained
reference metadata is never substituted for the current scene identity.

For the jaw workflow, connect the bound report to `3d.simple_part_instances`,
keep its legacy-binding option disabled, and supply the current depth capture
ID. The adapter checks equality and preserves the locator's binding details.
Accepted geometry still requires the jaw tool's independent current-depth,
contact and clearance checks; this contract does not qualify physical picking.
