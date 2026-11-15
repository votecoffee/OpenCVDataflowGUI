# Connection health

These data-only examples demonstrate disconnected status monitors. They do not
need an image source. The real-protocol templates start disabled and need no
plugin to open. Configure and enable them for the intended endpoint before use.

| Project | Check |
| --- | --- |
| [ROS 2](ros2_disabled.json) | Node/service discovery, read-only parameter-service response and image-topic receipt freshness |
| [MQTT](mqtt_disabled.json) | Broker CONNACK, subscription acknowledgment and optional message receipt |
| [EtherNet/IP](ethernetip_disabled.json) | Read-only CIP Identity |
| [ControlLogix](logix_disabled.json) | CIP Identity plus two named tag reads |
| [FANUC](fanuc_disabled.json) | Workbench controller status plus R[1] access |

The `192.0.2.1` addresses are documentation placeholders. The FANUC template
requires selection of a local Workbench profile. No PLC code, outputs, register
values or robot motions are changed. ROS response checking is limited to an
explicit read-only GetParameters call with empty names. Other service types use
discovery only. See [monitoring contracts](../../../docs/connection-health.md).

To see all three colors without connecting to equipment, first use **Load Python
plugin** to explicitly load this folder's [plugin.py](plugin.py), then open
[offline_ok.json](offline_ok.json), [offline_warning.json](offline_warning.json)
or [offline_alarm.json](offline_alarm.json). These authored fixtures are labeled
**Replay connection health (offline)** and cannot make network requests. They
also show how to implement another provider using `HealthProbe` and
`ConnectionHealth`. The alarm example deliberately marks its run as failed.

Select the monitor to expose optional status/report connectors. Open **Connection
diagnostics…** to see related application logs. With an enabled real endpoint,
use **Continuous** to keep scheduling probes. The thumbnail shows its last check
time and turns amber when that observation becomes stale. Stopping the job stops
probing; this is not an independent real-time watchdog.

From the repository root:

```powershell
python -m opencv_dataflow.runner "examples/industrial integration/connection_health/mqtt_disabled.json"
python -m opencv_dataflow.runner "examples/industrial integration/connection_health/offline_warning.json" --plugin "examples/industrial integration/connection_health/plugin.py"
python "examples/industrial integration/connection_health/build_example.py"
```

Code, graphs and authored fixture observations are MIT, by OpenCVDataflowGUI
contributors. No external images, device observations or third-party sample
records are redistributed. Every project includes an on-canvas scope Comment.
