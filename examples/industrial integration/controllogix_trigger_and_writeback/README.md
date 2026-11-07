# ControlLogix trigger and writeback

Open [controllogix_trigger_and_writeback.json](controllogix_trigger_and_writeback.json)
to inspect the wiring. Run once leaves the hardware and inspection branches
waiting or inactive. The saved PLC reader and writer are disabled and both
routes are blank.

| Example tag | Purpose |
| --- | --- |
| `Program:Vision.InspectRequest` | BOOL acquisition request. A rising edge releases one capture after the initial baseline. |
| `Program:Vision.PartSequence` | Sequence value used as the capture ticket. |
| `Program:Vision.ResultCount` | DINT destination for the inspected part count. |

The reader's **Ticket** gates the demo image source in **External** ticket
mode. The synthetic image contains two parts. The inspection assigns a final
verdict to the ticket and forwards the numeric count to **Write Logix tag**.
The writer also requires the reader's **Trigger** and its own **Write
enabled** setting. Inspect **Readback** for the verified value when using
an authorized test controller.

The addresses illustrate the wiring. Configure the actual route, tags,
and handshake for the cell using the [ControlLogix setup guide](../../../docs/industrial-integration.md#controllogix).
