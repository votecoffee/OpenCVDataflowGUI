# Industrial integration examples

| Project | What it shows |
| --- | --- |
| [FANUC hand–eye calibration](fanuc_hand_eye_calibration/README.md) | Run a calibration from synthetic FANUC XYZ/WPR captures and matching target poses. |
| [FANUC registers and frames](fanuc_registers_and_frames/README.md) | Inspect the wiring for R and PR reads, indexed UFRAME and UTOOL reads, flange pose composition, and a reviewed register write. |
| [ControlLogix trigger and writeback](controllogix_trigger_and_writeback/README.md) | Follow a PLC request through a real PCB photo inspection and result tag/readback; a separate offline replay needs no controller. |

The calibration runs offline. The two hardware templates open with controller
access disabled. See the [industrial integration guide](../../docs/industrial-integration.md)
for driver setup and operating details.


Real-image variants should include the acquisition data needed for the claim.
The PLC template now uses a real VisA photograph, with a local
[offline replay](controllogix_trigger_and_writeback/offline_photo_replay.json).
FANUC register/frame examples are data-only. The hand–eye example retains
controlled pose pairs for exact offline geometry; captured calibration requires
matched target photographs, intrinsics and robot poses. Source photographs
without that pairing do not qualify a hand–eye result.
