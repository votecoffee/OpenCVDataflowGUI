"""Create saved native execution projects with retained references and fixed fit settings."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession, make_registry

from opencv_simple_surface_enhance.dataflow import register
from opencv_simple_surface_enhance.recipe import import_capture, save_recipe


def build(studies, output):
    studies, output = studies.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    registry = make_registry()
    register(registry)
    trained = {}
    for key in ("analytic_clean", "polyhaven_clean", "four_light_ambiguity"):
        recipe = import_capture(studies / key / "capture.json")
        digest = save_recipe(assets / (key + ".sse"), recipe)
        trained[key] = digest
    checks = []
    for case in json.loads((studies / "summary.json").read_bytes())["cases"]:
        name = case["case"]
        training = (
            name
            if name == "four_light_ambiguity"
            else ("polyhaven_clean" if name.startswith("polyhaven") else "analytic_clean")
        )
        with ProjectSession(registry, directory=output) as session:
            session.apply(
                [
                    dict(
                        op="add",
                        type="image.simple_surface_enhance",
                        id="surface",
                        parameters=dict(
                            recipe_path=f"assets/{training}.sse",
                            sha256=trained[training],
                            source="capture manifest",
                            minimum_coverage=0.95,
                            capture_manifest=Path(
                                os.path.relpath(studies / name / "capture.json", output)
                            ).as_posix(),
                        ),
                        position=[0, 0],
                    ),
                    dict(
                        op="update",
                        id="surface",
                        fields=dict(
                            node_size="large",
                            thumbnail=True,
                            preview_output="overview",
                            description="Rendered diffuse control; frozen reference/calibration. "
                            "Same 95% availability threshold for all cases. "
                            "Not defect or gauge acceptance.",
                        ),
                    ),
                ]
            )
            assert session.inspect()["validation"]["valid"]
            session.save(output / (name + ".json"))
        with ProjectSession.open(output / (name + ".json"), registry) as reopened:
            run = reopened.run()
            block = reopened.graph.blocks["surface"]
            result = block.last_result
            assert result is not None, block.error
            expected_pass = case["coverage"] >= 0.95 and case["valid_pixels"] > 0
            assert run["passed"] == result.passed == expected_pass
            assert result.outputs["valid_pixels"] == case["valid_pixels"]
            with np.load(studies / name / "result/maps.npz", allow_pickle=False) as expected:
                for key in expected.files:
                    np.testing.assert_array_equal(result.outputs["maps"][key], expected[key])
            compiled = block.cache.estimator
            block.set_graphics_mode("off")
            assert reopened.run()["passed"] == expected_pass
            assert block.cache.estimator is compiled
            for key in result.outputs["maps"]:
                np.testing.assert_array_equal(
                    block.last_result.outputs["maps"][key], result.outputs["maps"][key]
                )
            checks.append(
                dict(
                    case=name,
                    training=training,
                    valid_pixels=case["valid_pixels"],
                    coverage=case["coverage"],
                    minimum_coverage=0.95,
                    expected_pass=expected_pass,
                    actual_pass=result.passed,
                    standalone_maps_exact=True,
                    graphics_off_maps_exact=True,
                    compiled_estimator_reused=True,
                    saved_reopened=True,
                    expected_judgment_matched=True,
                    scope="Rendered control, availability decision only",
                )
            )
    (output / "expected.json").write_bytes((json.dumps(checks, indent=2) + "\n").encode())
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = build(args.studies, args.output)
    print(json.dumps(dict(cases=len(rows), expected_judgments_matched=True)))
