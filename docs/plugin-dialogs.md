# Plugin dialogs

A trusted Python plugin can add a button to a block's **Configuration**
inspector. Set `plugin_dialog_label` on the block class and implement
`create_plugin_dialog(self, parent, parameters, context)`. Registration stays
the same. The desktop editor calls the factory on the Qt GUI thread when the
user clicks the button. Import Qt inside the factory if the rest of the plugin
can load and run independently of the GUI.

Return a fresh PySide6 `QDialog` built with the supplied `parent`. The host
opens it as a window-modal popup and owns its lifetime. The plugin can use any
Qt widgets, provide a read-only view, or implement its own editor and preview
workflow. [Qt dialog reference](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QDialog.html)

Set `plugin_dialog_modeless = True` to let users keep the popup open while
working in the graph or running it. The popup belongs to the selected block.
Changing the selection, leaving the inspector, or saving to a different
project directory closes it.

## Draft and apply

`parameters` is a private copy of the current inspector draft. Update the
dictionary before calling `dialog.accept()`. Keep the declared parameter names
and use JSON-compatible values of the corresponding types. For a patch, call
`parameters.update(patch)`. A review-only dialog can leave the draft untouched.

On **OK**, the host validates the whole draft and stages all edits together.
**Run block** and **Live results** can preview the staged values. **Apply
configuration** commits them through the usual inspector workflow. **Cancel**
or leaving the inspector discards popup edits. Starting a graph run closes a
window-modal popup. A modeless popup can stage edits while the graph runs using
the applied configuration. Apply staged edits after the run stops.

In a modeless popup, edits to different fields merge with the current
inspector draft. If the inspector and popup change the same field to different
values, the host discards the popup update and names the conflict. Factory,
validation, and conflict errors appear in **Diagnostics**.

Keep worker state in the dialog or plugin. Deliver worker results to the GUI
thread and discard stale results before editing the draft. Connect to the
dialog's `finished` signal to cancel outstanding work. Use `self` to read
block information and the draft dictionary to propose configuration changes.

## Project and image context

The `context` argument is a `PluginDialogContext` with the following members.

| Member | Purpose |
| --- | --- |
| `block_id` | ID of the block whose dialog opened. |
| `project_dir` | Directory of the current project, or `None` for an unsaved project. |
| `image_outputs` | Cached image outputs at opening or the last refresh. Each entry gives `block_id`, `block_name`, `port`, and `level_count`. |
| `refresh_images()` | Refreshes the catalog from the latest completed outputs published to the editor and returns the new entries. |
| `snapshot_image(block_id, port, *, level=0)` | Returns a private `ValuePacket` containing a copied image and copied wire payload. |
| `snapshot_input_image(port, *, source="connected", level=None)` | Copies an image for a named input of the selected block. |
| `output_revision` | Number of result publications received by the editor since the popup opened. Starts at zero. |
| `subscribe_outputs(callback)` | Calls `callback(revision)` on the GUI thread after a publication and returns an unsubscribe function. |

Run the graph in a profile that retains input and output previews, such as
Debug, to populate the cache. Use `refresh_images()` on the GUI thread
before taking a new snapshot in a modeless dialog. The catalog and snapshot API
are available for the life of the popup. Copies already taken can outlive it.
Select any cached image output and use a zero-based level for a pyramid, with
level zero as the largest. Snapshots retain pixel depth, channel encoding,
spatial and pyramid transforms, and ticket metadata. You can keep separate
reference and preview snapshots and process the copies in a plugin worker.
Store asset paths and IDs as normal block parameters if the graph needs them.

### Capture a block input

Call `context.snapshot_input_image("image")` to follow the wire into the
block. It uses the cached upstream output and the inspector's pyramid level
at popup opening. Pass `level=N` to select a different level. For inputs
that receive a whole pyramid, a capture defaults to its largest image.

Use `source="processed"` to capture the last input the block received.
That capture keeps the actual level, ticket, and outcome, even if a newer
image on the wire was rejected. It also works for injected inputs inside a
ToolBlock. When the block received a single image, this capture contains
that image at its original coordinates. When it received a whole pyramid,
you can select a level with `level=N`.

Both captures read the catalog at opening or the last `refresh_images()`.
The method raises a `ValueError` when the input image is unavailable.

## Follow graph outputs

Use `subscribe_outputs` when the popup should react to new graph results.
There is no initial callback. During continuous runs the editor publishes
results at the current run profile's display interval. Pending notices
collapse to the newest revision before dispatch. A notice carries the revision
number. The catalog and pixels are fetched separately, on demand.

```python
def outputs_ready(revision):
    status_label.setText(f"Graph output {revision} is ready")

unsubscribe = context.subscribe_outputs(outputs_ready)
```

Keep callbacks short. Refresh the catalog and copy an image on the GUI thread
when you need one, then hand the copy to a worker for heavier processing. For
live previews, compare the captured revision with `context.output_revision`
before showing a worker result. The host removes subscriptions on close.
Call the returned function to unsubscribe earlier. A callback that raises is
removed and logged in **Diagnostics**. Other subscribers continue.

## Add an image picker

Use `PluginImagePicker` from `opencv_dataflow.plugin_image_picker` inside a
Qt dialog. It refreshes its choices when the editor publishes results and
preserves the selected output and level across refreshes. If an output or
level becomes unavailable, choose another or wait for it to return.

```python
from opencv_dataflow.plugin_image_picker import PluginImagePicker

picker = PluginImagePicker(context, dialog, input_port="image")
picker.captured.connect(receive_image)
layout.addWidget(picker)
```

The optional `input_port` adds wired and last processed input shortcuts.
Those shortcuts use the input's normal level. Choose a graph output to
select its pyramid level explicitly. **Capture image** emits a private
`ValuePacket` through `captured`. Call `picker.snapshot()` to request the
same capture from your own controls. Each picker keeps its own selection, so
a dialog can use separate pickers for reference and inspection images.

See the [custom invert plugin](../examples/custom_invert/custom_invert.py)
for a working factory and [its project](../examples/custom_invert/README.md)
for the desktop and headless workflows.
