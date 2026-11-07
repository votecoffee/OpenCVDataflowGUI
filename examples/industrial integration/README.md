# Industrial integration examples

| Project | What it shows |
| --- | --- |
| [FANUC hand–eye calibration](fanuc_hand_eye_calibration/README.md) | Run a calibration from synthetic FANUC XYZ/WPR captures and matching target poses. |
| [FANUC registers and frames](fanuc_registers_and_frames/README.md) | Inspect the wiring for R and PR reads, indexed UFRAME and UTOOL reads, flange pose composition, and a reviewed register write. |
| [ControlLogix trigger and writeback](controllogix_trigger_and_writeback/README.md) | Follow a PLC request and part-sequence ticket through image acquisition, inspection, and a result tag with readback. |

The calibration runs offline. The two hardware templates open with controller
access disabled. See the [industrial integration guide](../../docs/industrial-integration.md)
for driver setup and operating details.
