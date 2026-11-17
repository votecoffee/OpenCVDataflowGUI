"""Saved metadata-only and conflicting-capture controls for the real locator chain."""

import argparse
import hashlib
import json
from pathlib import Path

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin


def check(session, *, mismatch):
    session.run()
    locator = session.graph.blocks["locate"]
    jaw = session.graph.blocks["jaw"]
    if mismatch:
        assert locator.last_result is None and locator.state == "error"
        assert "different captures" in locator.error
        assert locator._verified is None and locator._locator is None
        assert jaw.last_result is None
        return dict(
            locator_state=locator.state,
            reason=locator.error,
            model_not_loaded=True,
            matching_not_started=True,
            no_current_jaw_result=True,
            no_current_pose=True,
        )
    assert locator.last_result is not None, locator.error
    report = locator.last_result.outputs["report"]
    adapter = session.graph.blocks["instances"]
    assert adapter.last_result is not None, adapter.error
    assert report["capture_binding"]["method"] == "matching_cloud_metadata"
    assert adapter.parameters["allow_legacy_capture"] is False
    assert adapter.last_result.outputs["report"]["capture_binding"] == "embedded_report_capture_id"
    assert jaw.last_result is not None and jaw.last_result.outputs["count"] == 0
    return dict(
        locator_instances=len(report["matches"]),
        capture_id=report["capture_id"],
        method=report["capture_binding"]["method"],
        strict_adapter=True,
        eligible_grasps=0,
    )


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    source = root / "locator_chain/reconstructed/project.json"
    results = []
    for mismatch in (False, True):
        name = "mismatched_pair" if mismatch else "metadata_only"
        path = output / name / f"{name}.json"
        path.parent.mkdir()
        with ProjectSession.open(source, registry) as session:
            session.save(path)
            operations = [
                dict(
                    op="disconnect",
                    source="capture",
                    output="capture_id",
                    target="locate",
                    input="capture_id",
                )
            ]
            if mismatch:
                records = json.loads((path.parent / "assets/records.json").read_bytes())
                records["capture_id"] = "different_recorded_acquisition"
                (path.parent / "assets/other_capture.json").write_text(
                    json.dumps(records, indent=2) + "\n", encoding="utf-8"
                )
                operations += [
                    dict(
                        op="add",
                        id="other_capture",
                        type="input.simple_jaw_example",
                        parameters=dict(
                            depth_path="assets/depth.npy",
                            records_path="assets/other_capture.json",
                            locator_depth_path="assets/full.ply",
                        ),
                    ),
                    dict(
                        op="disconnect",
                        source="capture",
                        output="locator_depth",
                        target="locate",
                        input="depth_scene",
                    ),
                    dict(
                        op="connect",
                        source="other_capture",
                        output="locator_depth",
                        target="locate",
                        input="depth_scene",
                    ),
                ]
            session.apply(operations)
            assert session.inspect()["validation"]["valid"]
            session.save(path)
            result = check(session, mismatch=mismatch)
            results.append(dict(case=name, **result))
        with ProjectSession.open(path, registry) as reopened:
            assert check(reopened, mismatch=mismatch) == result
    (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


def verify_existing(root, output):
    """Check both saved controls and their relocated, byte-preserved assets."""
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    results = []
    for mismatch in (False, True):
        name = "mismatched_pair" if mismatch else "metadata_only"
        source = root / "capture_controls" / name / f"{name}.json"
        saved = output / name / f"{name}.json"
        saved.parent.mkdir()
        with ProjectSession.open(source, registry) as session:
            expected = check(session, mismatch=mismatch)
            session.save(saved)
        assets = {}
        for original in sorted((source.parent / "assets").iterdir()):
            assert original.is_file()
            copied = saved.parent / "assets" / original.name
            assert copied.read_bytes() == original.read_bytes(), original.name
            assets[original.name] = hashlib.sha256(original.read_bytes()).hexdigest()
        with ProjectSession.open(saved, registry) as relocated:
            assert relocated.inspect()["validation"]["valid"]
            assert check(relocated, mismatch=mismatch) == expected
        results.append(
            dict(case=name, **expected, portable_save_as_exact=True, assets_sha256=assets)
        )
    (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    action = verify_existing if args.verify_existing else build
    action(args.root.resolve(), args.output.resolve())
