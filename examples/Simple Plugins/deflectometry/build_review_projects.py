"""Native source/map/result loading reviews; file-source PASS is not estimator execution."""

import argparse
import json
from pathlib import Path

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from PIL import Image

VIEWS = (
    "overview",
    "overlay",
    "reference_source",
    "test_source",
    "reference_phase_x",
    "test_phase_x",
    "reference_phase_y",
    "test_phase_y",
    "change_x",
    "change_y",
    "magnitude",
    "amplitude_x",
    "amplitude_y",
    "residual_x",
    "residual_y",
    "quality",
)


def verify(studies):
    registry = make_registry()
    checks = []
    for row in json.loads((studies / "summary.json").read_bytes()):
        folder = studies / row["case"]
        with ProjectSession.open(folder / "project.json", registry) as session:
            assert session.inspect()["validation"]["valid"]
            assert session.run()["passed"]
            assert "cloud" not in session.graph.blocks
            for name in VIEWS:
                actual = session.graph.blocks[name].last_result.outputs["image"]
                expected = np.asarray(Image.open(folder / "result" / (name + ".png")))
                np.testing.assert_array_equal(actual, expected)
            checks.append(
                dict(
                    case=row["case"],
                    saved_reopened=True,
                    file_sources_passed=True,
                    estimator_reexecuted=False,
                    review_images_exact=len(VIEWS),
                    numeric_estimator_pass=row["actual_pass"],
                    metric_cloud=False,
                )
            )
    return checks


def build(studies):
    registry = make_registry()
    for row in json.loads((studies / "summary.json").read_bytes()):
        folder = studies / row["case"]
        if (folder / "project.json").exists():
            raise FileExistsError(folder / "project.json")
        actions = []
        for index, name in enumerate(VIEWS):
            actions += [
                dict(
                    op="add",
                    type="input.file",
                    id=name,
                    name=name.replace("_", " ").title(),
                    parameters=dict(
                        path=f"result/{name}.png", path_base="project", color_format="rgb"
                    ),
                    position=[(index % 3) * 460, (index // 3) * 340],
                ),
                dict(
                    op="update",
                    id=name,
                    fields=dict(node_size="large", thumbnail=True, preview_output="image"),
                ),
            ]
        actions += [
            dict(op="add", type="layout.comment", id="scope", position=[460, 1700]),
            dict(
                op="update",
                id="scope",
                fields=dict(
                    name="Saved-result review only",
                    description="PASS means image files loaded; no estimator runs. "
                    "Wrapped phase has whole-period ambiguity and no metric XYZ. "
                    + (
                        "SOFAST real exported capture compared with itself; "
                        "uncalibrated code diagnostics, CC BY-SA 4.0."
                        if row["case"] == "sofast_self_review"
                        else "Analytic phase-change/missing control, not physical shape truth."
                    ),
                ),
            ),
        ]
        with ProjectSession(registry, directory=folder) as session:
            session.apply(actions)
            assert session.inspect()["validation"]["valid"]
            session.save(folder / "project.json")
    checks = verify(studies)
    (studies / "host_review_checks.json").write_bytes(
        (json.dumps(checks, indent=2) + "\n").encode()
    )
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    checks = verify(args.studies.resolve()) if args.verify_only else build(args.studies.resolve())
    print(
        json.dumps(
            dict(
                cases=len(checks), exact_review_images=sum(c["review_images_exact"] for c in checks)
            )
        )
    )
