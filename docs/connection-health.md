# Connection health monitors

Five independent blocks live under **Industrial integration**, in the matching
ROS 2, MQTT, ControlLogix, EtherNet/IP or FANUC group:

| Block ID | Monitor |
| --- | --- |
| `industrial.ros2_health` | ROS 2 participant, configured nodes/services and topics |
| `industrial.mqtt_health` | Broker handshake and optional subscription/receipt checks |
| `industrial.ethernetip_health` | Read-only CIP Identity response |
| `industrial.logix_health` | CIP Identity plus optional named tag reads |
| `industrial.fanuc_health` | Workbench controller status plus optional register reads |

Configure the endpoint and enable the monitor. It needs no input connections and
runs alongside other graph roots, including when another branch has no input.
**Run once** checks it once; **Continuous** keeps scheduling graph activations.
The polling interval is a minimum interval between protocol probes. Intermediate
activations reuse the last result with `updated=false`, its original UTC check
time and a current age. This is graph scheduling, not an independent real-time
watchdog: stopped graphs do not probe, and acquisition waits or long iterations
can delay the next check. A thumbnail older than **Stale after seconds** turns
amber and says **STALE**. Downstream values update only when the graph executes.

## States and outputs

- **Green / OK:** the protocol check and all configured resource checks passed.
- **Yellow / Warning:** the endpoint answered but a configured tag, register,
  subscription, topic type/QoS match or receipt-freshness check failed.
- **Red / Alarm:** no valid protocol response, a runtime/driver cannot run, or a
  configured critical ROS node/service was not discovered. An unavailable driver
  or local configuration is distinguished by `error_type` and exception logs;
  red alone does not prove a remote device is powered off.
- **Gray / Disabled or Not checked:** monitoring is disabled or has not run.

The status thumbnail uses the last result directly, so it remains visible with
result graphics disabled and in Lean mode. It includes the last check time.
All connectors start hidden; expose **OK**, **Warning**, **Alarm**, **Connected**,
**State**, **Updated**, **Report** or **Logs** in the inspector as needed.
`connected=true` can accompany an alarm for a missing critical ROS service.
Reports contain per-resource reasons, protocol latency, UTC time, age and log
source paths. Latency measures the complete probe, potentially several requests.

**Fail on alarm** defaults off, so a disconnected status monitor does not fail
otherwise unrelated image inspections. Enable it to mark the monitor's run as
failed. To gate ticketed work or assign a final ticket judgment, connect its status
outputs to the relevant decision/judgment tools. Monitors do not create image
tickets. Warnings never fail the monitor automatically.

## Protocol checks

**ROS 2** uses the existing separately configured Lyrical/Fast DDS worker. ROS is
decentralized: with no remote expectations configured, green means the local
participant and observer executor are running. Required nodes use fully qualified
names; required services use `{ "name": "/driver/health", "type": "std_srvs/srv/Trigger" }`.
Missing required nodes/services are red. Service checking defaults to discovery.
For a standard parameter service, for example `/camera/driver/get_parameters`
with type `rcl_interfaces/srv/GetParameters`, set `"check_response": true` to
require a bounded, read-only response to an empty names request. An advertised
but unresponsive service then alarms while the local participant stays connected.
Pending requests are removed/canceled after the check window. No parameter values
are read or changed. Other service types use discovery only and cannot establish
response health; arbitrary Trigger, motion or application services are not called.
The [rclpy graph API](https://docs.ros.org/en/rolling/p/rclpy/api/node.html) defines
the discovery observations.

Topics use objects such as
`{ "name": "/camera/image", "type": "sensor_msgs/msg/Image", "minimum_publishers": 1, "max_age_seconds": 2 }`.
The observer checks actual type/QoS-compatible publisher matching and optional
local monotonic receipt age. Zero disables age checking. Receipt means a message
reached this observer, not that another consumer accepted it. Payloads are
discarded. Uninstalled topic interfaces and missing/incompatible/stale topics
are yellow while the observer remains responsive. Images and data use the same
subscription contract; no robot commands or simulation processes are started.

**MQTT** uses a fresh Paho connection and requires CONNACK for each probe, rather
than treating an open TCP port as broker health. Optional topic filters require
SUBACK. **Require message** also checks for a receipt within the probe window;
**Include retained** controls whether retained replay counts. Missing/rejected
subscriptions or missing qualifying messages warn. A failed connection or lost
transport alarms. No application messages are published. TLS and environment-based
passwords follow the existing MQTT blocks. See the
[Paho callbacks](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html).

**EtherNet/IP** opens an explicit CIP session and reads the Identity Object using
Get Attributes All. It does not set outputs, alter PLC code, or establish cyclic
I/O health. **ControlLogix** adds optional tag reads with pycomm3. A healthy tag
channel retains its metadata/client until reset or close. When tag access fails,
the monitor checks CIP Identity again: a responding endpoint gives yellow, a
failed heartbeat gives red. The failed tag channel reconnects on the next probe.
The optional `logix` dependency provides both adapters. See the
[pycomm3 CIP API](https://github.com/ottowayi/pycomm3/blob/master/pycomm3/cip_driver.py).

**FANUC** uses the configured Workbench profile and robot ID, checks a timestamped
controller-status response, and optionally reads named R/PR registers. On a
register fault it checks controller status again before choosing yellow. It
performs no motion, register write or service-management operation. Workbench's
normal permissions and retained read receipts still apply. Each request is
bounded by its timeout; checking many resources can require multiple requests.

## Aggregated diagnostics and extensions

Open **Connection diagnostics…** to view application logs from readers, writers
and monitors with the same endpoint configuration. This uses the existing severity,
search, pause, parameter-detail and exception-traceback viewer. The dialog performs
no network I/O. **Include logs** optionally emits up to twenty recent diagnostic
or exception events on the hidden Logs connector; it defaults off.

Grouping uses broker/credentials/TLS settings for MQTT, domain/discovery scope for
ROS, CIP route for Logix/EtherNet/IP and canonical Workbench profile/robot for FANUC.
These are endpoint groups, not a single shared socket. Monitoring uses its own
read-only probe; grouped peer errors are visible in logs but do not replace the
explicit resource checks. Configure the tags/topics/registers you want judged.
Log retention and Lean capture rules still apply. A successful probe does not
prove that a separate reader's private connection or payload processing is healthy.

Plugins can subclass `HealthProbe` (`check` and `close`) and `ConnectionHealth`
(`probe_factory`, parameters, stable type ID and protocol subtype). Checks return
JSON-compatible `connected`, `message` and bounded `resources`; each resource has
`name`, `state` (`ok`, `warning`, `alarm`) and `message`. Raise on transport failures.
Own and close handles, bound requests, observe cancellation and perform no writes.
Custom communication blocks can supply `connection_identity()` returning a
comparable endpoint key to join log groups. `bind_connection_sources` receives
graph peers when activity is bound. Runtime probes/handles are never serialized.
