# Industrial integration

The toolbox groups **ControlLogix** and **FANUC** tools under **Industrial
integration**. Online blocks start disabled and are unavailable in live preview.
Enable and configure them for the cell's intended endpoint before running a graph.

## ControlLogix

Install the optional pycomm3 driver in the application's Python environment.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[logix]"
```

The adapter uses explicit EtherNet/IP tag services, following the Python approach
used by OpenHMI. LogixTagRelay's implicit adapter is a separate interface for
cyclic I/O assemblies. [pycomm3 LogixDriver](https://pycomm3.readthedocs.io/en/stable/api_reference/logix_driver.html)

**Read Logix tags and trigger**

1. Configure the controller IP or CIP path, for example `192.0.2.1/0`, and
   enter the tag addresses in **Tags**. Program-scoped addresses use
   `Program:MainProgram.TagName`.
2. Optionally select a BOOL or integer **Trigger tag**. **Rising** waits for a
   low-to-high transition. **Change** emits when the value changes. Both first
   observe a baseline. **High** emits on each active poll. Without a trigger tag,
   each successful poll emits data.
3. The **Values** output contains the tag batch. **Value tag** exposes one
   address on the separate **Value** port. A numeric or string **Ticket tag**
   supplies the ticket value. Otherwise the normal source ticket settings apply.
4. Wire **Ticket** to an image source's **Ticket** input and set that source to
   **External** ticket mode to acquire only when the PLC reader emits. The
   **Trigger** output can enable the writeback for the same graph activation.

Continuous polling follows the graph execution cadence. Use a request held
until acknowledgement or a changing sequence number for reliable acquisition
when a PLC pulse could be shorter than the polling interval.

**Write Logix tag** takes one scalar value and a true **Enable** input. Select
the exact destination tag and turn on **Write enabled**. The writer checks that
the supplied scalar matches the destination's value type, performs one write,
and reports the readback. An uncertain write blocks subsequent writes in the
same runtime. Inspect the PLC value before resetting flow state. Use separate
activations or an application-level handshake when a cell needs multiple fields
updated together.

## FANUC

The online tools call the installed Windows [FANUC Workbench](https://github.com/votecoffee/FanucMCPServer)
through its JSON CLI. The Workbench profile must select the configured robot,
licensed FRRobot provider and allowed targets. Configure **Profile** and
**Robot ID** in each online block. DataflowGUI looks for Workbench 1.0.0 under
the normal per-user installation directory. Set the
`OPENCV_DATAFLOW_FANUC_HOME` environment variable to select another compatible
Workbench installation.

- **Read FANUC register** accepts `R[index]` or `PR[group,index]`. It returns
  the scalar or position record and the retained reference for a later review.
- **Read FANUC current position** returns the position for the selected motion
  group, including active user-frame and user-tool numbers. It reads the current
  position with frame IDs. Reading arbitrary UFRAME/UTOOL table entries and
  their offsets is outside this adapter's current interface.
- **Prepare FANUC register write** takes the retained reference, a true
  **Enable** input and a typed change such as `{"type":"integer","value":42}`
  for R or `{"values":{"X":120.5}}` for an initialized Cartesian PR. It returns
  a frozen review.
- **Commit approved FANUC register write** takes a true **Enable** input and a
  separately operator-approved object with `plan_id`, `review_sha256` and
  `approved_sha256`. Workbench verifies the current review, controller target,
  preimage and readback. Use Workbench's interactive console to commit if the
  local policy requires console confirmation. If a commit is interrupted, inspect
  the retained Workbench plan and controller before resetting flow state.

### Hand–eye samples

Wire a ticketed acquisition event to **Trigger** on the FANUC position reader,
then connect **Position** to **FANUC Cartesian pose**. The latter checks the
expected user-frame and user-tool numbers and converts XYZ/WPR to a rigid pose
in millimetres using `Rz(R) Ry(P) Rx(W)`. The pose retains the acquisition
ticket. For the default gripper-to-base labels, sample the flange in UF 0 and
UT 0. With other frames, name them explicitly and compose the known transforms
into the gripper-to-base convention before calibration.

Capture with the robot settled so the image and sequential robot read correspond
to the same pose. Batch the resulting gripper-to-base poses and matching
target-to-camera poses by ticket for **Calibrate hand–eye**.
