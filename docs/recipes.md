# Recipes and trained-model swaps

The job has one shared, versioned recipe catalog. Root blocks, nested ToolBlocks,
background acquisition and parallel part inspections use that catalog. This is a
job-wide system stored with the project, not a separate cross-process recipe
server or PLC driver.

A recipe contains an ID, revision, checksum, a `settings` JSON object and named
trained pattern/reference-image `models`. Numeric recipe requests are normalized to text IDs;
numeric `1` selects recipe `"1"`. Every saved revision is immutable. Updating
settings or training data creates another revision; older revisions remain
available for queued/in-flight work.

## Recipe manager

Open **Recipes > Manage recipes…** while stopped:

1. Enter a recipe ID and settings such as `{"threshold": 110, "min_score": 0.7}`.
2. Import sealed shape/texture/reference-image JSON, or select a trained pattern or Image memory tool and
   choose **Copy selected tool model…** (including a training block's last model output). Give each model an identifier such as
   `part`, `geometry` or `surface`. Model names cannot contain dots/spaces.
3. Click **Save new revision**. The first installed recipe becomes active and the
   project default. Later saves do not automatically activate the new revision.
4. Select a saved revision and choose **Activate saved revision**. The checkbox
   controls whether activation also changes the saved project default.
5. Save the project to retain recipes, models and bindings. Export/import recipe
   JSON for reuse; imports load editable content and require Save new revision.

Export preserves existing files rather than overwriting them. Import/copy changes
the editor's draft until saved. A recipe-bound matcher can also provide its last
selected asset for copying if it has no separately trained local model.

Manager edits and activation keep accepted queue data. They do not invalidate the
graph or rewrite existing recipe revisions. Use a unique part ticket for each
physical workpiece; recipe identity is separate from work/ticket identity.

Reference images use `trained_image_v1` assets. Extract a recipe slot such as
`models.reference` with **Recipe field** and wire it to Image memory's `reference`
input, alongside the same recipe-tagged image. See [sequence memory](sequence-memory.md)
for strict grid checks, readiness, asset limits and reference training controls.

## Recipe blocks

**Recipes > Selection > Select recipe** accepts an optional `recipe` data input
and optional `value` input. A request can be an installed ID, a revision reference,
or a full installed recipe snapshot. With no wired request, use `recipe_id` or the
value's pinned revision; an empty ID falls back to the active recipe.

Outputs are:

- `recipe`: the full selected immutable snapshot.
- `settings` and `models`: its JSON settings and named training assets.
- `value`: the supplied image/data, with a compact revision reference attached.
- `info`: ID, revision and checksum.

**Recipes > Outputs > Recipe value or model** extracts a dotted field, such as
`settings.threshold`, `settings.camera.exposure`, `settings.values.0`, or
`models.part`. Wire its data `value` output directly to a matcher's `model` input
to swap training data. It can accept the selected `recipe` snapshot, or an image
on its optional `value` input to use that image's pinned revision. Missing fields
are errors by default; `missing=inactive` suppresses the value output instead.

Known revision references are resolved exactly. Unknown IDs, missing revisions,
corrupt snapshots and incompatible pattern models are errors; the system does not
quietly substitute the current recipe/model. Shape and ORB/SIFT models are checked
using their existing sealed model schemas/checksums. Calibration profiles and
other JSON settings can be ordinary recipe fields; raw images/point clouds are
not recipe model assets in this version.

## Parameter and model bindings

The inspector's **Recipe overrides** section maps individual parameters to recipe
paths. For example, on Threshold set `value` to `settings.threshold`. On Gaussian
blur bind `kernel`, `sigma` or `border` to corresponding fields. Actual enum values
must be the names accepted by that block's existing dropdown.

Trainable matchers additionally have a **Trained model** binding, for example
`models.geometry`. This selects the stored shape/texture model per image without
adding a model-output wire. A wired `model` input and a model binding are competing
authorities and cannot be used together; choose one. The matcher rejects a wired
recipe-B model paired with a recipe-A image.

Effective settings are private to one invocation. The executor resolves the
image's revision, validates types/enums and native controls, runs the tool, then
restores configured parameters and any locally stored trained model even after an
error. The inspector result and headless receipt show the selected revision and
effective bound parameters. Runtime execution never overwrites the user's saved
tool defaults or training draft.

Overrides support processing/acquisition tools and compatible plugins. Buffered
flow scheduling, queue configuration, Counter input, Recipe blocks and composite
ToolBlock scheduling cannot be changed through these bindings. Bind processing
tools inside a ToolBlock or parallel inspection instead. Plugins can set
`recipe_parameters_supported = False` if their parameter lifecycle is unsuitable.
Training uses the tool's configured training controls; runtime recipe selection
does not retrain a model. Stored assets retain their detector/training settings,
so an ORB/SIFT swap is a model swap, not a change to an already-trained detector.

## Global activation and coordination

The manager changes the active recipe while stopped. A Select recipe block can
also set `activate_global=true` to activate a request at runtime. Runtime activation
does not change the saved project default. Use one ordered control writer;
selection blocks execute serially, and data dependencies establish when capture
occurs relative to the request. Two unrelated control branches are not a recipe
transaction protocol.

For PLC-style selection, feed its recipe ID to Select recipe's `recipe` input and
the part-counter value to `value`. Wire the selected `value` onward to a camera's
external `ticket` input. This orders selection before that acquisition and carries
the selected recipe reference with the trigger ticket. No PLC communication or
hardware-trigger driver is added by these blocks.

Image/depth/pyramid sources pin the active revision at invocation start, or use a
reference carried with their incoming trigger. A change while a native read is
running affects later invocations, not that invocation. Ordinary transforms,
queues and image pyramids preserve the compact `recipe_ref` payload.

An image already tagged with recipe A resolves A even if B is active. Reading a
pinned value never reactivates A globally, including when its captured recipe
snapshot is also wired to Select recipe. An explicit mismatching request is an
error unless `rebind=true`; rebinding is a deliberate relabeling of that value,
not a new capture or a global activation of already-pinned work.

When multiple sources must agree, coordinate their work tickets and recipe
selection explicitly. A latest-value cache of a global B model must not be used to
silently replace the A model for an older queued image. Either bind the matcher to
the image's recipe or derive Recipe value/model outputs from the pinned image or
its selected snapshot. Mixed-recipe groups are not one recipe execution context;
process their members separately or select a deliberately compatible context.
Ordinary multi-input processing, Join and ticket_all coordination reject differing
managed recipe revisions. Latest coordination can report `mixed_recipes: true`
for intentional previews; that does not make the group one inspection context.

Parallel part jobs inherit their parent image's recipe and share immutable assets.
Global activation inside parallel part inspections is rejected; it would otherwise
make results depend on worker completion order. Background capture uses the same
thread-safe catalog, while captured upstream snapshots preserve their version and
effective settings for downstream inspection.

## Persistence, provenance and limits

Project JSON stores recipe revisions once at the root and saves bindings on each
block. ToolBlock export includes catalog assets for portable reuse; embedding it
in a job merges catalogs, rejecting conflicting content for the same ID/revision.
Nested project definitions do not duplicate the shared bank. Runtime active state
survives Run once, Continuous and Stop/Resume; reloading starts with the saved
default. Images and buffered work are still runtime data, not saved project data.

`recipe_ref` contains only ID, revision and checksum, not a copy of the training
assets. Existing arbitrary payloads named `recipe` remain ordinary application
data and keep their previous behavior. Ticket events and receipts record the
managed reference; **Results > Ticket results…** can search recipe IDs/revisions.
This makes the actual recipe used for each ticket visible after a global switch.
A full recipe/model data output is ordinary snapshot data and increases queue
storage when captured upstream. Queue pinned images first and resolve model assets
downstream when practical. Captured effective overrides also count toward queue storage.

Limits are 16 MiB per serialized recipe and 128 total revisions / 64 MiB serialized
catalog content, plus runtime/GUI copy overhead. Capacity errors preserve existing
revisions. No revision is evicted automatically: queued images can still refer to
it. There is no revision-pruning UI, cross-job server, approval/release workflow or
automatic publication of freshly trained models yet. Install recipe content while
stopped; runtime selector blocks activate already installed revisions.

## Example

Open [recipes](../examples/flow_control/recipes/recipes.json). Widget A uses an edge model and ORB; Widget B
uses a corner model and SIFT, with different threshold/search settings. Both are
synthetic positive examples, not production acceptance thresholds. The geometric
model uses the inspector binding; the texture model comes from a Recipe value or
model block. Their results join by work ticket before final judgment.

Run once, open the manager, activate the other saved recipe, close the manager and
run again. Inspect model IDs, effective thresholds and the ticket recipe column.
Automated tests cover both recipes, model swaps, queued/background work, parallel
inspection, catalog persistence and GUI activation; no camera is required.

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/flow_control/recipes/recipes.json --repeat 3 --report .local/recipe-receipt.json
```
