"""Exercise the published native inspector and save three frozen teaching examples."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import cv2
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import render_cloud
from PySide6.QtWidgets import QApplication


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def settle(app, dialog):
    deadline = time.monotonic() + 30
    while dialog._future is not None and time.monotonic() < deadline:
        app.processEvents()
        dialog.poll()
        time.sleep(0.005)
    assert dialog._future is None, dialog.status.text()
    assert dialog.session.ready, dialog.status.text()


def build(root, output):
    from opencv_dataflow.app import MainWindow

    from opencv_simple_bin_picking.jaw_persistence import load_jaw_recipe
    from opencv_simple_bin_picking.jaw_teaching import reference_capture

    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    app = QApplication.instance() or QApplication([])
    records = []
    for case in ("clear", "obstacle", "tless/frame000013"):
        name = case.replace("/", "_")
        source = root / case / "project.json"
        destination = output / name / "project.json"
        destination.parent.mkdir(parents=True)
        with ProjectSession.open(source, registry) as session:
            session.save(destination)
            session.run()
            block = session.graph.blocks["jaw"]
            assert block.last_result is not None, block.error
            original = copy.deepcopy(block.parameters)
            before_count = block.last_result.outputs["count"]
            window = MainWindow()
            try:
                load_plugin(root / "plugin.py", window.registry)
                window.replace_graph(session.graph, destination)
                window.scene.nodes["jaw"].setSelected(True)
                app.processEvents()
                inspector = window.inspector
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                assert dialog.grasps.count() == 2 and dialog.editor.segments
                dialog.fields["workers"].setValue(2)
                dialog.reject()
                dialog.pool.shutdown(wait=True)
                assert block.parameters == original and inspector.read_parameters() == original
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                # Retain the supplied poses and raw depth through the actual connected context.
                dialog.capture_reference()
                capture = reference_capture(dialog.session.recipe)
                assert capture is not None and capture.identity["snapshot_id"]
                dialog.preview_reference()
                settle(app, dialog)
                assert len(dialog.session.result.picks) == before_count
                preview_report = dialog.session.result.report()
                dialog.stage()
                dialog.pool.shutdown(wait=True)
                assert block.parameters == original
                assert inspector.read_parameters()["recipe_path"] != original["recipe_path"]
                assert inspector.apply()
                staged_parameters = copy.deepcopy(block.parameters)
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                assert dialog.grasps.count() == 2
                assert reference_capture(dialog.session.recipe).capture_id == capture.capture_id
                dialog.preview_reference()
                settle(app, dialog)
                assert len(dialog.session.result.picks) == before_count
                dialog.reject()
                dialog.pool.shutdown(wait=True)
                assert block.parameters == staged_parameters
                session.save(destination)
            finally:
                window.dirty = False
                window.close()
                app.processEvents()
        with ProjectSession.open(destination, registry) as reopened:
            reopened.run()
            block = reopened.graph.blocks["jaw"]
            result = block.last_result
            assert result is not None, block.error
            report = result.outputs["report"]
            assert report["eligible_grasps"] == before_count
            assert [c["status"] for c in report["candidates"]] == [
                c["status"] for c in preview_report["candidates"]
            ]
            json_file(destination.with_name("expected.json"), report)
            cloud = result.outputs["graphics"]
            write_cloud(destination.with_name("result.ply"), cloud, binary=True)
            json_file(destination.with_name("graphics.json"), cloud.metadata)
            cv2.imwrite(
                str(destination.with_name("result.png")),
                render_cloud(cloud, 1000, 740, yaw=155, pitch=-35, point_size=2, limit=40000),
            )
            trained = load_jaw_recipe(
                destination.parent / block.parameters["recipe_path"],
                sha256=block.parameters["sha256"],
            )
            assert reference_capture(trained).capture_id == capture.capture_id
            assert trained.notes.get("license") == (
                "CC-BY-4.0" if case.startswith("tless") else None
            )
            records.append(
                dict(
                    case=name,
                    eligible=before_count,
                    inspector_cancel_preserved=True,
                    actual_connected_capture=True,
                    stage_did_not_apply=True,
                    inspector_apply_reopen=True,
                    retained_reference_preview=True,
                    expected_sha256=hashlib.sha256(
                        destination.with_name("expected.json").read_bytes()
                    ).hexdigest(),
                    assistance="Dataset annotation pose"
                    if case.startswith("tless")
                    else "Authored pose",
                )
            )
    json_file(output / "checks.json", records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Explicitly load before importing any version of BinPicking.
    initial = builtin_registry()
    load_plugin(args.root.resolve() / "plugin.py", initial)
    build(args.root.resolve(), args.output.resolve())
