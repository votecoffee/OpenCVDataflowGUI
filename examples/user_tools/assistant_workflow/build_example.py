"""Build a deterministic assistant/API/plugin demonstration with original synthetic pixels."""

import json
from pathlib import Path

from opencv_dataflow.assistant_api import ProjectSession, make_registry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLUGIN = ROOT / "skills/opencv-dataflow-plugins/assets/mean_inspection.py"


def operations():
    return [
        {"op": "add", "type": "input.demo", "id": "source", "name": "Synthetic target"},
        {"op": "add", "type": "example.mean_inspection", "id": "mean"},
        {
            "op": "update",
            "id": "mean",
            "fields": {
                "exposed_outputs": ["image", "mean", "passed", "result_graphics"],
                "description": "Accept the mean of selected pixels between 20 and 100 DN.",
                "documentation": "<p>Stateless example plugin with enum, numeric and coupled "
                "limits. A valid negative result carries a failure reason.</p>",
            },
        },
        {"op": "add", "type": "cv.overlay_result_graphics", "id": "review"},
        {"op": "add", "type": "inspect.final_judgment", "id": "judgment"},
        {"op": "add", "type": "layout.comment", "id": "notes"},
        {
            "op": "update",
            "id": "notes",
            "fields": {
                "name": "Assistant authoring example (MIT)",
                "description": (
                    "Original synthetic image from input.demo. No external image assets. "
                    "Load the mean_inspection.py skill template before opening this project. "
                    "The sibling operations.json and build_example.py show API authoring."
                ),
            },
        },
        {
            "op": "connect",
            "source": "source",
            "output": "image",
            "target": "mean",
            "input": "image",
        },
        {
            "op": "connect",
            "source": "mean",
            "output": "image",
            "target": "review",
            "input": "image",
        },
        {
            "op": "connect",
            "source": "mean",
            "output": "result_graphics",
            "target": "review",
            "input": "graphics_1",
        },
        {
            "op": "connect",
            "source": "mean",
            "output": "image",
            "target": "judgment",
            "input": "value",
        },
        {"op": "arrange"},
    ]


def build():
    actions = operations()
    (HERE / "operations.json").write_bytes((json.dumps(actions, indent=2) + "\n").encode())
    fail = [
        {
            "op": "update",
            "id": "mean",
            "fields": {"parameters": {"minimum": 200.0, "maximum": 255.0}},
        }
    ]
    (HERE / "fail_edits.json").write_bytes((json.dumps(fail, indent=2) + "\n").encode())
    with ProjectSession(make_registry([PLUGIN]), directory=HERE) as session:
        session.apply(actions)
        # Stable IDs and fresh destinations are also used when regenerating fixtures.
        content = json.dumps(session.graph.to_dict(), indent=2, allow_nan=False) + "\n"
        (HERE / "assistant_workflow.json").write_bytes(content.encode())


if __name__ == "__main__":
    build()
