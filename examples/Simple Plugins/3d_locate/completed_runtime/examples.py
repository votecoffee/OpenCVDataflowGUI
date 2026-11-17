"""Build and replay frozen native 3D reviews from unchanged public projects."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin

CASES = (
    ("real_positive", "real_tless/depth_verified"),
    ("real_no_match", "novel_models/object_000006"),
    ("generated_complete", ""),
    ("empty_control", "real_tless/depth_verified"),
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_result(result):
    def without_timing(value):
        if isinstance(value, dict):
            return {key: without_timing(item) for key, item in value.items() if key != "seconds"}
        if isinstance(value, list):
            return [without_timing(item) for item in value]
        return value

    report = without_timing(result.outputs["report"])
    return dict(
        passed=result.passed,
        report=report,
        poses=result.outputs["poses"],
        count=result.outputs["count"],
    )


def graphics_hashes(result):
    fit = result.outputs["fit"]
    return dict(
        points=hashlib.sha256(fit.points.tobytes()).hexdigest(),
        colors=hashlib.sha256(fit.colors.tobytes()).hexdigest(),
        overlay=None
        if result.outputs["overlay"] is NO_VALUE
        else hashlib.sha256(result.outputs["overlay"].tobytes()).hexdigest(),
    )


def native_review(app, window, block, export_parent=None):
    from opencv_simple_3d_locate import Locator, dataflow

    before = stable_result(block.last_result)
    charts = graphics_hashes(block.last_result)
    parameters = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    exports = []
    with (
        patch.object(Locator, "locate", side_effect=AssertionError("Localized")),
        patch.object(dataflow, "load_recipe", side_effect=AssertionError("Loaded recipe")),
        patch.object(block, "create_plugin_dialog", side_effect=AssertionError("Opened training")),
    ):
        inspector.open_plugin_review_dialog()
        app.processEvents()
        popup = inspector._plugin_review_dialog
        assert popup is not None and popup.snapshot is not None
        capture = popup.snapshot
        np.testing.assert_array_equal(popup.cloud.points, block.last_result.outputs["fit"].points)
        popup.yaw.setValue(35)
        popup.pitch.setValue(-25)
        if len(popup.cloud.points):
            popup.flyaround.setChecked(True)
            assert popup.orbit_timer.isActive()
            yaw = popup.yaw.value()
            popup.advance_orbit()
            assert popup.yaw.value() != yaw
            popup.flyaround.setChecked(False)
        options = [("all", None)]
        report = capture.record.report
        if report["matches"]:
            options.append(("accepted", 0))
        if report["execution"].get("candidate_review"):
            options.append(("rejected", 0))
        preferred = ("all", None)
        if export_parent is not None:
            if export_parent.name == "real_positive":
                preferred = ("accepted", 0)
            elif export_parent.name == "real_no_match":
                preferred = ("rejected", 0)
        for kind, index in options:
            popup.select_evidence(kind, index)
            popup.layers["scene"].setChecked(kind == "all")
            assert popup.image.pixmap() is not None
            if kind == "rejected":
                assert "not a detection" in popup.summary.text()
            if export_parent is not None and (kind, index) == preferred:
                name = kind if index is None else f"{kind}_{index}"
                popup.export_to(export_parent / name)
                exports.append((export_parent / name).as_posix())
        popup.select_evidence("input", "scene")
        np.testing.assert_array_equal(popup.cloud.points, capture.input_cloud().points)
        if "depth_scene" in capture.input_cloud_names:
            popup.select_evidence("input", "depth_scene")
            np.testing.assert_array_equal(
                popup.cloud.points, capture.input_cloud("depth_scene").points
            )
        assert (
            stable_result(block.last_result) == before
            and graphics_hashes(block.last_result) == charts
        )
        assert block.parameters == parameters
        popup.reject()
        app.processEvents()
    return capture, exports


def build(source, output):
    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.pointclouds import PointCloud
    from PySide6.QtWidgets import QApplication

    output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).resolve().parent
    for name in (
        "plugin.py",
        ".gitattributes",
        "README.md",
        "SOURCES.md",
        "LICENSE",
        "COMPLETED_REVIEW.md",
        "wheel-verification.json",
        "examples.py",
    ):
        shutil.copy2(template / name, output / name)
    shutil.copytree(template / "releases", output / "releases")
    for name, relative in (("object29", "real_tless"), ("object6", "novel_models")):
        folder = output / "sources" / name
        folder.mkdir(parents=True)
        for filename in ("ATTRIBUTION.md", "provenance.json", "selection.json"):
            original = source / relative / "source" / filename
            if original.is_file():
                shutil.copy2(original, folder / filename)
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    expected, rows = {}, []
    app = QApplication.instance() or QApplication([])
    for case, relative in CASES:
        folder = output / case
        folder.mkdir()
        original = source / relative
        shutil.copy2(original / "project.json", folder / "project.json")
        with ProjectSession.open(original / "project.json", registry) as session:
            paths = set()
            for block in session.graph.blocks.values():
                paths.update(block.project_assets())
                if block.type_id == "input.cloud":
                    paths.add(block.parameters["path"])
            for path in paths | {"assets/truth.json"}:
                (folder / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original / path, folder / path)
        path = folder / "project.json"
        with ProjectSession.open(path, registry) as session:
            block = session.graph.blocks["locate"]
            if case == "empty_control":
                write_cloud(
                    folder / "assets/empty.ply",
                    PointCloud(np.empty((0, 3)), "mm", "camera"),
                    binary=True,
                )
                session.graph.blocks["scene"].parameters["path"] = "assets/empty.ply"
            block.preview_output = "fit"
            session.save(path)
            assert session.inspect()["validation"]["valid"]
            session.run()
            assert block.state in ("pass", "fail"), block.error
            expected[case] = stable_result(block.last_result)
            window = MainWindow()
            warnings = []
            window.warn = warnings.append
            load_plugin(output / "plugin.py", window.registry)
            window.replace_graph(session.graph, path)
            window.show()
            try:
                captured, exports = native_review(app, window, block, output / "snapshots" / case)
                rows.append(
                    dict(
                        case=case,
                        count=block.last_result.outputs["count"],
                        state=captured.record.report["assessment"]["status"],
                        source_sha256=hashlib.sha256(
                            captured.input_cloud().points.tobytes()
                        ).hexdigest(),
                        recipe_sha256=hashlib.sha256(captured.recipe_bytes).hexdigest(),
                        graphics=graphics_hashes(block.last_result),
                        native_results_flyaround=True,
                        no_search_verification_reload_or_training=True,
                        primary_outputs_and_graphics_unchanged=True,
                        exports=[Path(p).relative_to(output).as_posix() for p in exports],
                    )
                )
                assert not warnings, warnings
            finally:
                window.dirty = False
                window.close()
                app.processEvents()
    write_json(output / "expected.json", expected)
    write_json(output / "native-verification.json", rows)
    print(
        json.dumps(
            dict(
                projects=len(rows),
                exports=sum(len(r["exports"]) for r in rows),
                counts=[row["count"] for row in rows],
            )
        )
    )


def verify(root, output, *, native=False, core_only=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    if core_only:
        from opencv_simple_3d_locate.dataflow import register

        registry.types.pop("3d.simple_locate", None)
        register(registry)
    else:
        load_plugin(root / "plugin.py", registry)
    app = None
    if native:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    expected = json.loads((root / "expected.json").read_bytes())
    originals = {
        row["case"]: row for row in json.loads((root / "native-verification.json").read_bytes())
    }
    rows, export_count = [], 0
    for row in originals.values():
        for relative in row["exports"]:
            folder = root / relative
            report = json.loads((folder / "review.json").read_bytes())
            assert (
                hashlib.sha256((folder / "executed_recipe.s3l").read_bytes()).hexdigest()
                == row["recipe_sha256"]
            )
            assert (
                hashlib.sha256(np.load(folder / "scene_points.npy").tobytes()).hexdigest()
                == row["source_sha256"]
            )
            assert not report["verification_reexecuted"] and not report["training_changed"]
            export_count += 1
    for case, _ in CASES:
        path = root / case / "project.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.run()
            block = session.graph.blocks["locate"]
            assert block.last_result is not None, block.error
            assert stable_result(block.last_result) == expected[case], case
            assert graphics_hashes(block.last_result) == originals[case]["graphics"], case
            if app is not None:
                from opencv_dataflow.app import MainWindow

                window = MainWindow()
                load_plugin(root / "plugin.py", window.registry)
                window.replace_graph(session.graph, path)
                window.show()
                try:
                    captured, _ = native_review(app, window, block)
                    assert (
                        hashlib.sha256(captured.input_cloud().points.tobytes()).hexdigest()
                        == originals[case]["source_sha256"]
                    )
                finally:
                    window.dirty = False
                    window.close()
                    app.processEvents()
            if not core_only:
                if case == "real_positive":
                    from opencv_simple_3d_locate import dataflow

                    render = block.parameters["render_overlay"]
                    block.parameters["render_overlay"] = False
                    with patch.object(dataflow, "overlay", side_effect=AssertionError("Rendered")):
                        session.run()
                    assert stable_result(block.last_result) == expected[case]
                    quiet = graphics_hashes(block.last_result)
                    assert quiet["overlay"] is None
                    assert all(
                        quiet[key] == originals[case]["graphics"][key]
                        for key in ("points", "colors")
                    )
                    block.parameters["render_overlay"] = render
                target = output / case / "project.json"
                target.parent.mkdir()
                recipe_name = block.parameters["recipe_path"]
                raw = (path.parent / recipe_name).read_bytes()
                session.save(target)
                assert (target.parent / recipe_name).read_bytes() == raw
        if not core_only:
            with ProjectSession.open(target, registry) as copied:
                copied.run()
                result = copied.graph.blocks["locate"].last_result
                assert result is not None and stable_result(result) == expected[case]
                assert graphics_hashes(result) == originals[case]["graphics"]
        rows.append(
            dict(
                case=case,
                numeric_exact=True,
                default_graphics_exact=True,
                native_review=native,
                save_as_reopen=not core_only,
            )
        )
    write_json(
        output / "checks.json",
        dict(
            core_only=core_only,
            cases=rows,
            exports_verified=export_count,
            scope="Execution seconds excluded; no independent accuracy qualification",
        ),
    )
    print(
        json.dumps(
            dict(
                projects=len(rows),
                exports=export_count,
                numeric_exact=True,
                default_graphics_exact=True,
                native=native,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "build":
        build(args.source.resolve(), args.output.resolve())
    else:
        verify(
            args.source.resolve(),
            args.output.resolve(),
            native=args.native,
            core_only=args.core_only,
        )
