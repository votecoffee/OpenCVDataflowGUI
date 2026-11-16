# Self-starting ROS image and data demos

Configure the local [ROS 2 Lyrical runtime](../../../docs/ros2.md) once. Open one
of these programs and press **Run once**:

- [Managed image demo](managed_image.json): a program-owned ROS publisher streams
  a real PCB photograph; the graph receives RGB pixels, converts to gray and finds
  edges. The source image is visible in the block results; image values are checked
  against the original fixture.
- [Managed data demo](managed_data.json): receives a JSON String with sequence,
  part and source fields. Its sequence becomes the acquisition ticket.

The first run starts the publisher through the configured Lyrical interpreter.
Further Run once executions reuse it; **Continuous** keeps receiving. **Stop**
pauses graph cycles. **New/Open**, closing the application, or **Program → Settings
and scripts → Runtime → Stop program services** destroys the owned ROS nodes and
closes the publisher. Opening/validating either project starts nothing. No separate
terminal, simulator, robot or hardware connection is needed.

Both use Fast DDS, localhost discovery, domain **71**, and private
`/opencv_dataflow/program_demo/image` and `/opencv_dataflow/program_demo/data`
topics. Avoid running both programs simultaneously in the same domain if source
uniqueness matters. Use a separate domain/topic in Program service arguments and
block parameters when needed. This demo does not launch the global ROS CLI daemon.

**Program → Settings and scripts** shows the enabled lifecycle configuration:
startup logging, a per-cycle counter/status, shutdown logging and a managed
service. The service waits for `ROS_DEMO_READY`; EOF asks its executor/node/context
to close. Output ends with `ROS_DEMO_STOPPED`. Timeouts provide an owned-tree cleanup
fallback. **Save As** copies declared script/image/source-record assets.

The photo is **VisA / SPot-the-Difference**, Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer, ECCV 2022, **CC BY 4.0**. The
[publisher record](https://registry.opendata.aws/visa/),
[license](https://creativecommons.org/licenses/by/4.0/), existing
[source/change record](../../assets/showcase/SOURCES.md) and [SOURCES.json](SOURCES.json)
identify the image and hashes. This reuses the resized/letterboxed PCB2 0000 PNG,
converted losslessly to RGB8 PPM; no additional pixel edits or defects. Both
canvases include attribution. Code and project definitions are MIT; image rights
remain CC BY 4.0.

```powershell
.venv/Scripts/opencv-dataflow-api.exe validate "examples/industrial integration/ros2_program_lifecycle/managed_image.json"
.venv/Scripts/opencv-dataflow-api.exe run "examples/industrial integration/ros2_program_lifecycle/managed_image.json" --repeat 2 --summary
```

See [program lifecycle contracts](../../../docs/program-lifecycle.md) for script
context/state, process ownership, failure handling and the assistant API.
