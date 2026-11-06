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

## Draft and apply

`parameters` is a private copy of the current inspector draft. Update the
dictionary before calling `dialog.accept()`. Keep the declared parameter names
and use JSON-compatible values of the corresponding types. For a patch, call
`parameters.update(patch)`. A review-only dialog can leave the draft untouched.

On **OK**, the host validates the whole draft and stages all edits together.
**Run block** and **Live results** can preview the staged values. **Apply
configuration** commits them through the usual inspector workflow. **Cancel**,
leaving the inspector, or starting a graph run discards popup edits. A factory
or validation failure appears in **Diagnostics** and leaves the previous
inspector draft in place.

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
| `image_outputs` | Cached image outputs in the current graph when the popup opened. Each entry gives `block_id`, `block_name`, `port`, and `level_count`. |
| `snapshot_image(block_id, port, *, level=0)` | Returns a private `ValuePacket` containing a copied image and copied wire payload. |

Run the graph to populate the cache before opening a dialog that needs images.
Select any cached image output and use a zero-based level for a pyramid, with
level zero as the largest. Snapshots retain pixel depth, channel encoding,
spatial and pyramid transforms, and ticket metadata. You can keep separate
reference and preview snapshots and process the copies in a plugin worker.
Store asset paths and IDs as normal block parameters if the graph needs them.

See the [custom invert plugin](../examples/custom_invert/custom_invert.py)
for a working factory and [its project](../examples/custom_invert/README.md)
for the desktop and headless workflows.
