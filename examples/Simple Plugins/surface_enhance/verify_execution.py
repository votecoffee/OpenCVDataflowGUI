"""Explicitly load the trusted plugin and reopen/run the five saved execution projects."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin


def verify(root, plugin):
    registry = make_registry()
    load_plugin(plugin.resolve(), registry)
    rows = json.loads((root / "execution/expected.json").read_bytes())
    checks = []
    for row in rows:
        name = row["case"]
        with ProjectSession.open(root / "execution" / (name + ".json"), registry) as session:
            assert session.inspect()["validation"]["valid"]
            block = session.graph.blocks["surface"]
            assert block.cache.recipe is None and block.last_result is None
            assert session.run()["passed"] == row["expected_pass"]
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["valid_pixels"] == row["valid_pixels"]
            assert len(result.graphics) == 8 and result.outputs["overview"].shape[2] == 3
            with np.load(root / "studies" / name / "result/maps.npz", allow_pickle=False) as maps:
                for key in maps.files:
                    np.testing.assert_array_equal(result.outputs["maps"][key], maps[key])
            compiled = block.cache.estimator
            block.set_graphics_mode("off")
            assert session.run()["passed"] == row["expected_pass"]
            assert block.cache.estimator is compiled
            assert block.last_result.outputs["overview"] is NO_VALUE
            assert not block.last_result.graphics
            assert block.last_result.outputs["report"] == result.outputs["report"]
            for key in result.outputs["maps"]:
                np.testing.assert_array_equal(
                    block.last_result.outputs["maps"][key], result.outputs["maps"][key]
                )
            checks.append(
                dict(
                    case=name,
                    opened_without_processing=True,
                    expected_pass=row["expected_pass"],
                    actual_pass=result.passed,
                    valid_pixels=row["valid_pixels"],
                    raw_maps_exact=True,
                    graphics_off_maps_exact=True,
                    compiled_estimator_reused=True,
                    rgb_result_views=8,
                    expected_judgment_matched=True,
                )
            )
    return dict(
        scope="Rendered controls; actual plugin estimation, not acquired-camera accuracy",
        plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
        cases=checks,
        expected_judgments_matched=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    checks = verify(args.root.resolve(), args.plugin.resolve())
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(dict(cases=len(checks["cases"]), expected_judgments_matched=True)))
