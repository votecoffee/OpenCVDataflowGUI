# Custom Python invert block

This example uses a custom Python block to invert a horizontal band in a synthetic
image. It exposes both the edited image and the original.

1. Use **File → Open…** to open [`custom_invert.json`](custom_invert.json), then click
   **Run once**. DataflowGUI loads the bundled example block automatically.
2. Select **Invert** and hover over **Start Percent** or **End Percent** to see the
   allowed range of 0–100. The project starts at 20% and ends at 80%.
3. Enter `-1` in **Start Percent** to see a range error. Enter `90` to see the
   rule that Start Percent must be lower than End Percent. The second error
   highlights both fields.
4. Set Start Percent to `30`. The warning clears. Click **Run block** to preview
   the change, or **Apply configuration** to keep it in the project.
5. Click **Edit inversion band…** to edit both percentages with the plugin's own
   controls. **OK** stages the values in the inspector. **Cancel** discards the
   popup edits. Click **Apply configuration** to keep accepted edits.
6. Keep the popup open and click **Capture image**. Click **Run
   once** in the main window. The popup reports when the next output is ready.
   Capture again to see the previous and latest captures with their tickets.
   The picker starts on the wire into **Invert**, using its configured input
   level. You can also choose the last processed input or another graph
   output and its pyramid level.

The [example source](../../../src/opencv_dataflow/example_plugins/invert.py) declares the
limits with `NumericRange` and raises `ValidationError`
with both field names for the cross-field rule. The inspector and runtime use
the same rules.
[Plugin dialog API](../../../docs/plugin-dialogs.md)

To practice manual plugin loading, select [`custom_invert.py`](custom_invert.py)
with **File → Load Python plugin…**. It registers the same bundled example block.

From the repository root, the example also runs headlessly.

```powershell
opencv-dataflow-run examples/user_tools/custom_invert/custom_invert.json
```

[All examples](../../README.md)
