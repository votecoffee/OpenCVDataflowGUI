"""Build actual execution projects with retained references and one availability gate."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.payloads import NO_VALUE

from opencv_simple_fringe_scan.dataflow import register
from opencv_simple_fringe_scan.recipe import import_capture, save_recipe


def build(studies, output):
    studies, output = studies.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir()
    registry = make_registry()
    register(registry)
    trained = {}
    for key in ("analytic_clean", "warsaw_crown_phase"):
        recipe = import_capture(studies / key / "capture.json")
        trained[key] = save_recipe(output / "assets" / (key + ".sfs"), recipe)
    rows = []
    for case in json.loads((studies / "summary.json").read_bytes()):
        name, expected = case["case"], case["report"]
        training = "analytic_clean" if name.startswith("analytic") else name
        with ProjectSession(registry, directory=output) as session:
            session.apply(
                [
                    dict(
                        op="add",
                        type="3d.simple_fringe_scan",
                        id="fringe",
                        position=[0, 0],
                        parameters=dict(
                            recipe_path=f"assets/{training}.sfs",
                            sha256=trained[training],
                            source="capture manifest",
                            minimum_coverage=0.95,
                            capture_manifest=Path(
                                os.path.relpath(studies / name / "capture.json", output)
                            ).as_posix(),
                        ),
                    ),
                    dict(
                        op="update",
                        id="fringe",
                        fields=dict(
                            node_size="large",
                            thumbnail=True,
                            preview_output="cloud" if expected["metric"] else "overview",
                            description=(
                                "Retained sequence/calibration. Uniform 95% coverage gate. "
                                "Rendered controls; camera accuracy remains unqualified."
                            ),
                        ),
                    ),
                ]
            )
            assert session.inspect()["validation"]["valid"]
            session.save(output / (name + ".json"))
        with ProjectSession.open(output / (name + ".json"), registry) as reopened:
            run = reopened.run()
            block = reopened.graph.blocks["fringe"]
            result = block.last_result
            assert result is not None, block.error
            passed = expected["valid_pixels"] > 0 and expected["coverage"] >= 0.95
            assert result.passed == run["passed"] == passed
            assert result.outputs["count"] == expected["valid_pixels"]
            with np.load(studies / name / "result/maps.npz", allow_pickle=False) as maps:
                for key in maps.files:
                    np.testing.assert_array_equal(result.outputs["maps"][key], maps[key])
            if expected["metric"]:
                cloud = result.outputs["cloud"]
                assert cloud.units == "m" and cloud.frame == "camera" and cloud.colors is not None
                assert len(cloud.points) == expected["valid_pixels"]
            else:
                assert result.outputs["cloud"] is NO_VALUE
            cached = block.cache.model
            block.set_graphics_mode("off")
            assert reopened.run()["passed"] == passed and block.cache.model is cached
            for key in result.outputs["maps"]:
                np.testing.assert_array_equal(
                    block.last_result.outputs["maps"][key], result.outputs["maps"][key]
                )
            rows.append(
                dict(
                    case=name,
                    training=training,
                    expected_pass=passed,
                    actual_pass=result.passed,
                    accepted=expected["valid_pixels"],
                    coverage=expected["coverage"],
                    minimum_coverage=0.95,
                    metric=expected["metric"],
                    standalone_maps_exact=True,
                    graphics_off_maps_exact=True,
                    compiled_model_reused=True,
                    saved_reopened=True,
                    expected_judgment_matched=True,
                )
            )
    (output / "expected.json").write_bytes((json.dumps(rows, indent=2) + "\n").encode())
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checks = build(args.studies, args.output)
    print(json.dumps(dict(cases=len(checks), expected_judgments_matched=True)))
