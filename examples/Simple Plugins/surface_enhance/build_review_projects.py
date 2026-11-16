"""Create/reopen file-only image-map reviews using the host's saved-project API."""

import argparse
import json
from pathlib import Path

from opencv_dataflow.assistant_api import ProjectSession, make_registry


def build(root):
    registry = make_registry()
    cases = json.loads((root / "summary.json").read_bytes())["cases"]
    reviews = []
    for case in cases:
        folder = root / case["case"]
        operations = []
        for index, name in enumerate(
            ("overview", "source", "normals", "albedo", "relight", "quality", "residual", "detail")
        ):
            operations.extend(
                [
                    dict(
                        op="add",
                        type="input.file",
                        id=name,
                        name=name.replace("_", " ").title(),
                        parameters=dict(
                            path=f"result/{name}.png", path_base="project", color_format="rgb"
                        ),
                        position=[(index % 4) * 410, (index // 4) * 330],
                    ),
                    dict(
                        op="update",
                        id=name,
                        fields=dict(node_size="large", thumbnail=True, preview_output="image"),
                    ),
                ]
            )
        operations.extend(
            [
                dict(op="add", type="layout.comment", id="scope", position=[0, 660]),
                dict(
                    op="update",
                    id="scope",
                    fields=dict(
                        name="Rendered-control saved review",
                        description="File-source PASS means loading only; no estimation runs. "
                        "Diffuse-render reference inputs, not captured multi-light images. "
                        "Normal/albedo evidence, not measured XYZ/height. " + case["source"],
                    ),
                ),
            ]
        )
        with ProjectSession(registry, directory=folder) as session:
            session.apply(operations)
            assert session.inspect()["validation"]["valid"]
            session.save(folder / "project.json")
        with ProjectSession.open(folder / "project.json", registry) as reopened:
            assert reopened.run()["passed"]
            for block in reopened.graph.blocks.values():
                if block.type_id == "input.file":
                    assert block.last_result.outputs["image"].dtype.name == "uint8"
        reviews.append(
            dict(
                case=case["case"],
                validated=True,
                saved_and_reopened=True,
                file_sources_passed=True,
                estimation_reexecuted=False,
            )
        )
    with (root / "host_review_checks.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(reviews, indent=2) + "\n")
    return reviews


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    print(json.dumps(build(parser.parse_args().studies)))
