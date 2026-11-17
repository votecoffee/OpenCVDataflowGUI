"""Explicitly load the trusted fringe plugin and replay its three execution graphs."""

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
            block = session.graph.blocks["fringe"]
            assert block.cache.recipe is None and block.last_result is None
            assert session.run()["passed"] == row["expected_pass"]
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["count"] == row["accepted"]
            views = 8 if row["metric"] else 6
            assert len(result.graphics) == views
            assert result.outputs["overview"].shape[2] == 3
            assert block.preview_output == ("cloud" if row["metric"] else "overview")
            with np.load(root / "studies" / name / "result/maps.npz", allow_pickle=False) as maps:
                assert set(maps.files) == set(result.outputs["maps"])
                for key in maps.files:
                    np.testing.assert_array_equal(result.outputs["maps"][key], maps[key])
            cloud = result.outputs["cloud"]
            if row["metric"]:
                assert cloud.units == "m" and cloud.frame == "camera"
                assert len(cloud.points) == row["accepted"] and cloud.colors is not None
                assert cloud.source_size == (96, 72)
                expected = result.outputs["maps"]
                selected = expected["geometry_valid"]
                np.testing.assert_array_equal(
                    cloud.points, expected["geometry_xyz_m"][selected].astype(np.float32)
                )
                np.testing.assert_array_equal(
                    cloud.pixels, expected["geometry_camera_pixels"][selected]
                )
            else:
                assert cloud is NO_VALUE
            compiled = block.cache.model
            block.set_graphics_mode("off")
            assert session.run()["passed"] == row["expected_pass"]
            assert block.cache.model is compiled
            after = block.last_result
            assert after.outputs["overview"] is NO_VALUE and not after.graphics
            assert after.outputs["report"] == result.outputs["report"]
            for key in result.outputs["maps"]:
                np.testing.assert_array_equal(
                    after.outputs["maps"][key], result.outputs["maps"][key]
                )
            if row["metric"]:
                assert after.outputs["cloud"].colors is None
                np.testing.assert_array_equal(after.outputs["cloud"].points, cloud.points)
                np.testing.assert_array_equal(after.outputs["cloud"].pixels, cloud.pixels)
            else:
                assert after.outputs["cloud"] is NO_VALUE
            checks.append(
                dict(
                    case=name,
                    opened_without_processing=True,
                    accepted=row["accepted"],
                    actual_pass=result.passed,
                    expected_pass=row["expected_pass"],
                    raw_maps_exact=True,
                    graphics_off_maps_points_report_exact=True,
                    compiled_model_reused=True,
                    rgb_result_views=views,
                    native_metric_cloud=row["metric"],
                    expected_judgment_matched=True,
                )
            )
    return dict(
        scope="Actual plugin execution on rendered inputs; no acquired-camera accuracy proof",
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
