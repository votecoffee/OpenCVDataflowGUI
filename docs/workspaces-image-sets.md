# Workspaces and image sets

## Groups and navigation

Rubber-band select nodes, then use **Workspace > Group selected nodes…** (Ctrl+G).
Each node belongs to at most one visual group. The dashed border and translucent
background follow member extents as individual nodes move. Drag the group header
or border to move all members and their wires. Group interiors remain available
for selecting nodes and connections. Select a group and press Delete to remove
only its container, or use **Ungroup selection** (Ctrl+Shift+G) with the group or
any member selected. These operations preserve processing results and connections.
Groups currently have one level; reusable processing hierarchy uses ToolBlocks.

The scene expands to include distant nodes. Wheel zoom supports large layouts,
middle-drag and scrollbars pan, and Fit graph frames all nodes. The upper-right
overview draws inexpensive node rectangles and a highlighted viewport. Click or
drag it to navigate without rendering image previews there. The dotted grid
adapts at low zoom to keep drawing costs bounded.

Save stores group membership, absolute node positions, view center and zoom in
project JSON. Nested ToolBlocks have independent workspace state. Creating a
ToolBlock transfers fully selected groups into its internal graph; partially
selected groups keep their remaining outer members. Navigation and grouping
mark the project modified, without clearing results. There is no automatic save.

## Image file sequences

The Image file inspector offers three `source_mode` choices:

- `single`: repeatedly load `path`, preserving the ordinary Run once workflow.
- `files`: load the ordered `paths` list, one image per activation. **Select images…**
  fills it from a multi-selection file picker; the order returned by that picker
  is preserved and can be edited in the JSON field.
- `directory`: snapshot supported image filenames in `directory`, sorted by
  case-insensitive full path with a deterministic tie break. `recursive` includes
  subdirectories. Unrelated extensions are ignored. Newly added files enter the
  sequence after resetting flow state or changing source configuration.

Apply configuration after choosing files or a directory. Existing `color_format`
and `pixel_depth` conversions apply to every image. `path_base=project` resolves
relative paths beside the saved project. Reading bytes before OpenCV decoding
supports Unicode filenames. The `info` output includes filename and path.

`repeat=stop` ends the sequence; Continuous stops automatically when all finite
sources finish. Background acquisition drains accepted queued frames before
stopping. `repeat=loop` repeats until Stop. Run once advances one entry; Stop/Resume
keeps the runtime cursor. Reset flow state or a configuration change restarts it.
Cursors and directory snapshots are runtime state: reopening a project starts
from the first image. Incomplete batches/joins may remain pending at end of set;
this does not invent a verdict for unfinished work. Configure matching source
counts and buffering policies for coordinated multi-source jobs.

`filename_ticket=true` uses the basename, including extension, as the ticket value
in automatic-number mode. Every occurrence has a distinct internal ticket session,
so duplicate names, repeated paths and loops do not merge inspection records.
Explicit fixed, external/PLC and timestamp ticket modes take precedence; disable
filename tickets to retain ordinary auto-number values. Corrupt files receive an
error for their filename ticket and the next activation advances to the next file.
Invalid/empty source selections stop with a source diagnostic.

Run the included set without Qt:

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/inputs/image_set_workspace/image_set_workspace.json --all-images
```

It intentionally produces one pass and two failures, so the CLI returns inspection
failure status (1). `--all-images` requires at least one finite file/directory
source. Ordinary `--repeat N` also stops when the sequence is exhausted.

## Filmstrip

**Results > Image results filmstrip** toggles the bottom dock. File-source runs
show it automatically. Each source image has a thumbnail, filename and labeled
border: green pass, red fail, purple execution error/conflicting final judgments,
amber pending. Borders follow ticket final judgments or completed job outcomes,
not the source block's successful decoding. Pending entries update as buffered
branches finish. Hover to see the path, ticket value and failure reasons. A broken
image has an error placeholder. Validation sets have their own filmstrip: a
correctly predicted failing image still has a red border, independently of its
validation score.

Capture keeps at most 200 thumbnails and 8 MiB of encoded thumbnail bytes; oldest
entries are evicted. Full-resolution images and filmstrip histories are not saved
in projects or validation JSON reports. GUI publication sampling does not lose
entries because workers capture into a shared, thread-safe store. **Clear** clears
the retained thumbnails, independently of production counts and ticket history.

Thumbnail conversion adds work. **Capture result thumbnails** defaults on for
Debug/Balanced and off for Lean. Set it before starting a run; it is sampled at run
start. Lean can explicitly opt in. Headless runs do not generate thumbnails unless
an API caller supplies/enables a filmstrip. The validation window has its own
capture checkbox.
