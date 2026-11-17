"""Build four one-block execution projects, sharing the clean analytic training."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession, make_registry

from opencv_simple_deflectometry.dataflow import register
from opencv_simple_deflectometry.recipe import import_reference, save_recipe


def build(studies, output):
    studies, output = studies.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir()
    registry = make_registry()
    register(registry)
    trained = {}
    rows = []
    for row in json.loads((studies / "summary.json").read_bytes()):
        training = "reference" if row["case"].startswith("analytic") else "sofast_reference"
        if training not in trained:
            recipe = import_reference(studies / row["reference"], maximum_change_rad=0.45)
            trained[training] = save_recipe(output / "assets" / (training + ".sdef"), recipe)
        manifest = Path(os.path.relpath(studies / row["capture"], output)).as_posix()
        path = output / (row["case"] + ".json")
        with ProjectSession(registry, directory=output) as session:
            session.apply(
                [
                    dict(
                        op="add",
                        type="image.simple_deflectometry",
                        id="deflect",
                        parameters=dict(
                            recipe_path=f"assets/{training}.sdef",
                            sha256=trained[training],
                            source="capture manifest",
                            capture_manifest=manifest,
                        ),
                    ),
                    dict(
                        op="update",
                        id="deflect",
                        fields=dict(
                            node_size="large",
                            thumbnail=True,
                            preview_output="overlay",
                            description="Actual phase-reference execution. "
                            "Same generic settings/95% coverage/0.45-rad limit in all cases. "
                            "SOFAST is same-capture code diagnostics; no metric height.",
                        ),
                    ),
                ]
            )
            assert session.inspect()["validation"]["valid"]
            session.save(path)
        with ProjectSession.open(path, registry) as session:
            assert session.run()["passed"] == row["expected_pass"]
            block = session.graph.blocks["deflect"]
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["count"] == row["comparable"] and len(result.graphics) == 16
            with np.load(studies / row["case"] / "result/maps.npz", allow_pickle=False) as baseline:
                for key in baseline.files:
                    np.testing.assert_array_equal(result.outputs["maps"][key], baseline[key])
            assert json.loads(json.dumps(result.outputs["report"])) == json.loads(
                (studies / row["case"] / "result/report.json").read_bytes()
            )
            model = block.cache.model
            block.set_graphics_mode("off")
            assert session.run()["passed"] == row["expected_pass"] and block.cache.model is model
            assert block.last_result.outputs["report"] == result.outputs["report"]
            for key, value in result.outputs["maps"].items():
                np.testing.assert_array_equal(block.last_result.outputs["maps"][key], value)
        rows.append(
            dict(
                case=row["case"],
                training=training,
                expected_pass=row["expected_pass"],
                comparable=row["comparable"],
                coverage=row["coverage"],
                minimum_coverage=0.95,
                maximum_change_rad=0.45,
                raw_maps_and_report_exact=True,
                graphics_off_exact=True,
                compiled_reference_reused=True,
                saved_reopened=True,
            )
        )
    (output / "expected.json").write_bytes((json.dumps(rows, indent=2) + "\n").encode())
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            dict(cases=len(build(args.studies, args.output)), expected_judgments_matched=True)
        )
    )
