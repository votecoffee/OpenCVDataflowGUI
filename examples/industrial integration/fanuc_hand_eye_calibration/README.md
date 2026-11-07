# FANUC hand–eye calibration

Open [fanuc_hand_eye_calibration.json](fanuc_hand_eye_calibration.json) and
click **Run once**. The project uses four synthetic FANUC position records
and matching synthetic target-to-camera poses. It runs offline with built-in
blocks.

Double-click a **Saved capture** ToolBlock to see its capture ticket, XYZ/WPR
record, and **FANUC Cartesian pose** block. The converter checks UF 0 and
UT 0 and emits a flange-to-base pose in millimetres. Each target pose carries
the same ticket as its corresponding flange pose. The two batches feed
**Calibrate hand–eye**, which pairs the samples by ticket.

Inspect **Camera to gripper** on the fit block. The known mount translation
is `[40, -25, 90]` mm. **Report** shows the four capture tickets and residuals.
For live acquisition wiring and conversion from nonzero user frames and
tools, see the [FANUC registers and frames template](../fanuc_registers_and_frames/README.md).
