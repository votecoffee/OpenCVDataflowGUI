"""Offline provider demonstration. No networking, hardware, secrets or external assets."""

from opencv_dataflow.connection_health import ConnectionHealth, HealthProbe


class RecordedProbe(HealthProbe):
    def check(self, parameters, project_dir, cancelled):
        state = parameters["recorded_state"]
        connected = state != "alarm"
        return {
            "connected": connected,
            "message": "Recorded offline protocol observation",
            "resources": []
            if not connected
            else [
                {
                    "name": "example/topic_or_tag",
                    "state": state,
                    "message": "Recorded healthy resource"
                    if state == "ok"
                    else "Recorded missing subscription/tag; endpoint remains responsive",
                }
            ],
        }


class RecordedHealth(ConnectionHealth):
    type_id, title = "example.recorded_connection_health", "Replay connection health (offline)"
    subtype = "Offline health demonstration"
    probe_factory = RecordedProbe
    defaults = {**ConnectionHealth.defaults, "enabled": True, "recorded_state": "ok"}
    parameter_choices = {"recorded_state": ("ok", "warning", "alarm")}
    parameter_help = {
        **ConnectionHealth.parameter_help,
        "recorded_state": "Authored offline fixture. It is not a network observation.",
    }


def register(registry):
    registry.register(RecordedHealth)
