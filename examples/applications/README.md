# Application examples

Prefer real, openly licensed captures for application demonstrations. The
following are useful real-data starting points; each linked project includes
its source/license and any setup requirements.

| Application | Data and project |
| --- | --- |
| PCB presence/outline | [VisA photographs](pcb_outline/README.md). |
| Photographic color comparison | [Lab matching](lab_color_match/README.md) and [palette classification](lab_color_classification/README.md) on a real VisA PCB. |
| Shipping-label decoding | [Dynamsoft real label photographs](shipping_label_barcodes/README.md). |
| Weld-bead inspection | [Real weld photographs](<../Simple Plugins/path_width/weld_bead/README.md>), using the optional SimplePathWidth integration. Pixel widths have no calibrated physical-width truth. |
| PCB pose/localization | [Real VisA pattern example](<../Simple Plugins/pattern_matching/visa_pcb2/README.md>), using SimplePatternMatch. |
| Clear/occupied tote | [Real ARMBench reference inspection](<../Simple Plugins/reference_inspect/armbench_tote/README.md>), with fixed-view assumptions. |
| Photographic defect comparison | [VisA reference-inspection hard cases](<../Simple Plugins/reference_inspect/visa_pcb2/README.md>), retaining known false alarms and misses. |

The [full applications index](../README.md#applications) also retains controlled
fixtures for exact bead/contour geometry, generated Data Matrix payloads,
known pose offsets and simulated conveyor/encoder motion. Those are explicitly
synthetic evidence for a rule or coordinate contract. Real conveyor tracking and
robot calibration need captures paired with encoder/robot data; a photograph
alone cannot supply those measurements. New showcase variants should lead with
appropriate real captures as those inputs become available.

See the [real-data source policy](../README.md#image-policy) and
[Simple Plugins guide](<../Simple Plugins/README.md>).
