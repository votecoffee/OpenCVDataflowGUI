"""Explicit final ticket judgments, distinct from ordinary tool pass/fail."""

from .core import Block, Port, Result
from .outcomes import failed_input
from .payloads import ValuePacket


class FinalJudgment(Block):
    type_id, title = "inspect.final_judgment", "Final ticket judgment"
    category, block_type, subtype = "Inspection", "Inspection", "Ticket results"
    inputs = (Port("value", "any"), Port("passed", "data", required=False))
    outputs = (Port("value", "any"), Port("judgment", "data"))
    defaults = {"mode": "upstream", "data": {}}
    parameter_choices = {"mode": ("upstream", "boolean_input", "pass", "fail")}
    parameter_help = {
        "mode": "Upstream uses accumulated status. Other modes assign an explicit final decision.",
        "data": "Compact JSON record attached to the final ticket event.",
    }
    preserve_image_sets = True
    final_ticket_input = "value"

    def process(self, inputs):
        payload = self.input_payloads["value"]
        if "ticket" not in payload:
            raise ValueError("Final judgment requires a ticketed value")
        mode = self.parameters["mode"]
        if mode == "boolean_input":
            if type(inputs.get("passed")) is not bool:
                raise ValueError("Boolean input mode requires a boolean passed input")
            passed = inputs["passed"]
        else:
            passed = (
                (not failed_input(ValuePacket(inputs["value"], payload)))
                if mode == "upstream"
                else mode == "pass"
            )
        return Result(
            {"value": inputs["value"], "judgment": {"passed": passed, "final": True}},
            passed=passed,
            final_judgment=passed,
            event_data=self.parameters["data"],
            message=f"Final {'PASS' if passed else 'FAIL'} ({mode})",
        )
