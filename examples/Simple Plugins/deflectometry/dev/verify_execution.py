"""Load the explicit trusted plugin and replay all four sealed-recipe execution graphs."""

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
    checks = []
    for row in json.loads((root / "execution/expected.json").read_bytes()):
        with ProjectSession.open(root / "execution" / (row["case"] + ".json"), registry) as session:
            assert session.inspect()["validation"]["valid"]
            block = session.graph.blocks["deflect"]
            assert block.cache.model is None and block.last_result is None
            assert session.run()["passed"] == row["expected_pass"]
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["count"] == row["comparable"] and len(result.graphics) == 16
            assert result.outputs["overlay"].shape[2] == 3 and block.preview_output == "overlay"
            with np.load(
                root / "studies" / row["case"] / "result/maps.npz", allow_pickle=False
            ) as baseline:
                for key in baseline.files:
                    np.testing.assert_array_equal(result.outputs["maps"][key], baseline[key])
            assert json.loads(json.dumps(result.outputs["report"])) == json.loads(
                (root / "studies" / row["case"] / "result/report.json").read_bytes()
            )
            model = block.cache.model
            block.set_graphics_mode("off")
            assert session.run()["passed"] == row["expected_pass"] and block.cache.model is model
            lean = block.last_result
            assert lean.outputs["report"] == result.outputs["report"] and not lean.graphics
            assert lean.outputs["overlay"] is NO_VALUE and lean.outputs["overview"] is NO_VALUE
            for key, value in result.outputs["maps"].items():
                np.testing.assert_array_equal(lean.outputs["maps"][key], value)
        checks.append(
            dict(
                case=row["case"],
                comparable=row["comparable"],
                expected_pass=row["expected_pass"],
                actual_pass=result.passed,
                opened_without_processing=True,
                saved_reopened=True,
                raw_maps_and_report_exact=True,
                graphics_off_exact=True,
                compiled_reference_reused=True,
                rgb_result_images=16,
                metric_height=False,
            )
        )
    return dict(
        plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
        cases=checks,
        expected_judgments_matched=True,
        scope="Actual plugin execution, not physical accuracy",
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
    args.receipt.write_bytes((json.dumps(checks, indent=2) + "\n").encode())
    print(json.dumps(dict(cases=len(checks["cases"]), expected_judgments_matched=True)))
