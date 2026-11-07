# FANUC registers and frames

Open [fanuc_registers_and_frames.json](fanuc_registers_and_frames.json) to
inspect a FANUC wiring template. Run once leaves the hardware branches
waiting or inactive. The saved read and write controls are disabled and
the Workbench profile and robot ID are blank.

The upper branch samples current position, then wires its active frame and
tool numbers into the indexed UFRAME and UTOOL readers. The Cartesian
converter expects **UF 3** and **UT 2**. The composition chain applies the
inverse UTOOL, the measured TCP pose, and the UFRAME offset in that order
to express the flange pose in robot-base coordinates. This pose is the
robot-side input for eye-in-hand calibration. Capture the robot position
and camera observation while the robot is settled.

The lower branch shows a numeric register read at example address `R[10]`
and a position register read at `PR[1,10]`. The R read's retained reference
goes to **Prepare review for R[10]** with an example typed change. Both
write stages use a false interlock. The commit input's placeholder produces
no approval. A live commit requires a separately approved Workbench review
and a cell-specific enable signal.

Use the [FANUC setup guide](../../../docs/industrial-integration.md#fanuc)
to configure an authorized robot and select appropriate register addresses.
