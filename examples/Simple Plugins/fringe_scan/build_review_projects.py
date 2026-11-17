"""Native saved-result map/cloud review; file-source success is loading only."""

import argparse
import json
from pathlib import Path

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession, make_registry


def verify_project(folder, row, registry):
    with ProjectSession.open(folder / "project.json", registry) as session:
        assert session.inspect()["validation"]["valid"]
        assert session.run()["passed"]
        if row["report"]["metric"]:
            cloud = session.graph.blocks["cloud"].last_result.outputs["cloud"]
            assert len(cloud.points) == row["report"]["valid_pixels"]
            assert cloud.units == "m" and cloud.frame == "camera" and cloud.colors is not None
            with np.load(folder / "result/maps.npz", allow_pickle=False) as maps:
                np.testing.assert_allclose(
                    cloud.points, maps["geometry_xyz_m"][maps["geometry_valid"]], atol=1e-7, rtol=0
                )
        else:
            assert "cloud" not in session.graph.blocks
    return dict(
        case=row["case"],
        saved_reopened=True,
        file_sources_passed=True,
        reconstruction_reexecuted=False,
        native_cloud_points=row["report"]["valid_pixels"] if row["report"]["metric"] else 0,
        native_flyaround=bool(row["report"]["metric"]),
    )


def build(studies):
    registry = make_registry()
    rows = json.loads((studies / "summary.json").read_bytes())
    checks = []
    for row in rows:
        folder = studies / row["case"]
        actions = []
        for index, name in enumerate(row["graphics_views"]):
            actions += [
                dict(
                    op="add",
                    type="input.file",
                    id=name,
                    name=name.title(),
                    parameters=dict(
                        path=f"result/{name}.png", path_base="project", color_format="rgb"
                    ),
                    position=[(index % 3) * 420, (index // 3) * 330],
                ),
                dict(
                    op="update",
                    id=name,
                    fields=dict(node_size="large", thumbnail=True, preview_output="image"),
                ),
            ]
        if row["report"]["metric"]:
            actions += [
                dict(
                    op="add",
                    type="input.cloud",
                    id="cloud",
                    name="Accepted metric rays / native fly-around",
                    parameters=dict(
                        path="result/accepted.ply", units="m", frame="camera", max_points=1000000
                    ),
                    position=[840, 660],
                ),
                dict(
                    op="update",
                    id="cloud",
                    fields=dict(node_size="large", thumbnail=True, preview_output="cloud"),
                ),
            ]
        actions += [
            dict(op="add", type="layout.comment", id="scope", position=[0, 660]),
            dict(
                op="update",
                id="scope",
                fields=dict(
                    name="Saved-result loading review",
                    description="File-source PASS means loading only; no decoder runs. "
                    + (
                        "Analytic first-surface metric control, not physical accuracy evidence."
                        if row["report"]["metric"]
                        else "CC BY4 Wlasiuk/Sitnik SIMULATED phase scan; no metric cloud."
                    ),
                ),
            ),
        ]
        with ProjectSession(registry, directory=folder) as session:
            session.apply(actions)
            assert session.inspect()["validation"]["valid"]
            session.save(folder / "project.json")
        checks.append(verify_project(folder, row, registry))
    (studies / "host_review_checks.json").write_bytes(
        (json.dumps(checks, indent=2) + "\n").encode()
    )
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    print(json.dumps(build(parser.parse_args().studies.resolve())))
