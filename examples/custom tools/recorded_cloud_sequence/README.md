# Replay recorded point clouds

Open [recorded_cloud_sequence.json](recorded_cloud_sequence.json) and select
**Run All**. The three scans run in the order of the [manifest](assets/recording.json).
The filenames intentionally sort in a different order.

| Scan | Format | Recorded frame | Units | Nanoseconds from the example epoch |
| --- | --- | --- | --- | ---: |
| `z_entrance.ply` | Binary PLY | `conveyor` | mm | 3 |
| `middle.pcd` | Binary PCD | `conveyor` | m | 45,000,007 |
| `a_review.obj` | ASCII OBJ | `tool_camera` | m | 105,000,017 |

The **Info** output reports the exact integer timestamp on the
`scanner_clock` clock and the zero-based scan index. **Batch serial entries**
buffers all three scans before producing a group. **Unbatch serial entries**
emits the three original packets with their own recorded timestamps, frames,
units, and tickets. Where a file carries source IDs, replay keeps them.
Otherwise **Track source rows** adds IDs using the file order.

In the editor, **Run All** rewinds and stops after the final scan. **Continuous**
loops the sequence. In headless runs, use `--all-images` with **Repeat** set
to `stop`. Each graph cycle consumes one scan, so the capture timestamps
remain available for analysis independently of the processing rate.

A manifest uses the `opencv_dataflow_cloud_recording_v1` format, a named
`clock`, and a `scans` list. Each record has `path`, `timestamp_ns`,
`frame`, and `units` (`m`, `mm`, or `unitless`). Scan paths stay within
the manifest folder and use forward slashes. Timestamps are strictly
increasing nonnegative signed 64-bit integers. The block accepts up to
20,000 entries in 8 MiB and applies **Max points** to each scan read.
An unreadable or empty scan, or file metadata that disagrees with the
manifest, stops playback before emitting that scan or issuing its ticket.

The scans and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
