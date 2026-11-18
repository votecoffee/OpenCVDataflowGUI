"""Build/replay completed bin-picking reviews from unchanged public recipes."""

import argparse
import copy
import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import PointCloud

CASES = (
    ("suction_good", "generated/good"),
    ("suction_holes", "generated/holes"),
    ("suction_missing", "generated/missing"),
    ("suction_real13", "real_tless/frame000013"),
    ("jaw_clear", "known_part_jaw/clear"),
    ("jaw_obstacle", "known_part_jaw/obstacle"),
    ("jaw_unknown", "known_part_jaw/unknown"),
    ("jaw_real3", "known_part_jaw/tless/frame000003"),
    ("jaw_real13", "known_part_jaw/tless/frame000013"),
    ("jaw_mesh", "known_part_jaw/mesh_training/clear"),
    ("jaw_graphics_off", "known_part_jaw/clear"),
)
EXPORTS = {"suction_good", "jaw_clear", "jaw_obstacle", "jaw_unknown", "jaw_real13", "jaw_mesh"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_result(result):
    return json.loads(
        json.dumps(
            dict(
                passed=bool(result.passed),
                score=result.score,
                message=result.message,
                fail_reasons=result.fail_reasons,
                outputs={
                    name: value
                    for name, value in result.outputs.items()
                    if value is not NO_VALUE and not isinstance(value, PointCloud)
                },
            ),
            allow_nan=False,
        )
    )


def graphics_hashes(result):
    return {
        name: {
            member: hashlib.sha256(getattr(value, member).tobytes()).hexdigest()
            for member in ("points", "colors")
        }
        for name, value in result.outputs.items()
        if isinstance(value, PointCloud)
    }


def no_replanning():
    from opencv_simple_bin_picking import dataflow, jaw_dataflow
    from opencv_simple_bin_picking.jaw import JawPlanner
    from opencv_simple_bin_picking.planning import Picker
    from opencv_simple_bin_picking.scene import Scene

    stack = ExitStack()
    for owner, name in (
        (Picker, "plan"),
        (JawPlanner, "plan"),
        (Scene, "__init__"),
        (Scene, "sample"),
        (Scene, "contact"),
        (dataflow, "load_recipe"),
        (jaw_dataflow, "load_jaw_recipe"),
    ):
        stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
    stack.enter_context(
        patch(
            "opencv_simple_bin_picking.jaw_raster.render_depth",
            side_effect=AssertionError("Projected depth"),
        )
    )
    return stack


def native_review(app, window, block, target=None):
    before, graphics = stable_result(block.last_result), graphics_hashes(block.last_result)
    parameters = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    with (
        no_replanning(),
        patch.object(block, "create_plugin_dialog", side_effect=AssertionError("Opened training")),
    ):
        inspector.open_plugin_review_dialog()
        app.processEvents()
        popup = inspector._plugin_review_dialog
        assert popup.snapshot is not None
        frozen = popup.snapshot
        emitted = block.last_result.outputs["graphics"]
        if emitted is not NO_VALUE:
            np.testing.assert_array_equal(popup.cloud.points, emitted.points)
            np.testing.assert_array_equal(popup.cloud.colors, emitted.colors)
        popup.flyaround.setChecked(True)
        assert popup.orbit_timer.isActive()
        yaw = popup.yaw.value()
        popup.advance_orbit()
        assert popup.yaw.value() != yaw
        candidates = frozen.record.report["candidates"]
        chosen = candidates[0] if candidates else None
        if chosen is not None:
            popup.select_candidate(chosen["id"])
            popup.focus_candidate()
            assert not popup.layers["scene"].isChecked()
            popup.layers["witnesses"].setChecked(False)
            popup.layers["witnesses"].setChecked(True)
            # Export context as well as the selected candidate.
            popup.layers["scene"].setChecked(True)
            popup.layers["bin"].setChecked(True)
        if target is not None:
            popup.export_to(target)
        popup.mode.setCurrentIndex(popup.mode.findData("observations"))
        assert popup.cloud.metadata["scope"] == "retained_measured_first_returns"
        for name in ("pose_fit", "mesh_fit"):
            index = popup.mode.findData(name)
            if frozen.report()["diagnostic_arrays"][name]:
                assert index >= 0
                popup.mode.setCurrentIndex(index)
            else:
                assert index < 0
        popup.overview()
        if emitted is not NO_VALUE:
            np.testing.assert_array_equal(popup.cloud.points, emitted.points)
        assert block.parameters == parameters
        assert stable_result(block.last_result) == before
        assert graphics_hashes(block.last_result) == graphics
        popup.reject()
        assert popup.snapshot is None and popup.cloud is None and popup._rendered is None
        app.processEvents()
    return frozen


def picker(graph):
    return next(
        block
        for block in graph.blocks.values()
        if block.type_id in ("3d.simple_bin_picking", "3d.simple_bin_picking_jaw")
    )


def build(source, output):
    from opencv_dataflow.app import MainWindow
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
    sources = output / "sources"
    sources.mkdir()
    for original, name in (
        (source / "known_part_jaw/tless/provenance.json", "jaw-provenance.json"),
        (source / "real_tless/normalization.json", "suction-normalization.json"),
    ):
        shutil.copy2(original, sources / name)
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    app = QApplication.instance() or QApplication([])
    projects = output / "projects"
    projects.mkdir()
    assets = projects / "assets"
    assets.mkdir()
    expected, rows = {}, []
    for case, relative in CASES:
        original, path = source / relative, projects / f"{case}.json"
        shutil.copy2(original / "project.json", path)
        with ProjectSession.open(path, registry) as session:
            for block in session.graph.blocks.values():
                names = (
                    ("path",) if block.type_id == "input.file" else block.project_asset_parameters
                )
                for name in names:
                    value = block.parameters.get(name)
                    if not value:
                        continue
                    assert isinstance(value, str)
                    current = original / value
                    raw = current.read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    destination = assets / f"{digest}{current.suffix}"
                    if destination.exists():
                        assert destination.read_bytes() == raw
                    else:
                        destination.write_bytes(raw)
                    block.parameters[name] = f"assets/{destination.name}"
            block = picker(session.graph)
            if case == "jaw_graphics_off":
                block.parameters["render_graphics"] = False
            block.description = (
                "Review completed bin-picking evidence in the inspector. "
                "Recorded first surfaces; no robot execution or physical grasp truth."
            )
            session.save(path)
            assert session.inspect()["validation"]["valid"], case
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
                target = output / "snapshots" / case if case in EXPORTS else None
                if target is not None:
                    target.parent.mkdir(exist_ok=True)
                frozen = native_review(app, window, block, target)
                report = frozen.record.report
                rows.append(
                    dict(
                        case=case,
                        passed=block.last_result.passed,
                        eligible=len(frozen._result.picks),
                        candidates=len(report["candidates"]),
                        decisions=sorted({c["status"] for c in report["candidates"]}),
                        source_sha256=frozen.input_description("depth")["sha256"],
                        recipe_sha256=hashlib.sha256(frozen.recipe_bytes).hexdigest(),
                        graphics=graphics_hashes(block.last_result),
                        diagnostics=frozen.report()["diagnostic_arrays"],
                        native_flyaround=True,
                        no_replanning_or_projection=True,
                    )
                )
                assert not warnings, warnings
            finally:
                window.dirty = False
                window.close()
                app.processEvents()
    write_json(output / "expected.json", expected)
    write_json(
        output / "verification.json",
        dict(
            cases=rows,
            projects=len(rows),
            frozen_exports=len(EXPORTS),
            scope="Offline evidence; no physical grasp qualification",
        ),
    )
    write_json(
        output / "integrity.json",
        {
            str(path.relative_to(output)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.suffix != ".pyc"
        },
    )
    print(json.dumps(dict(projects=len(rows), exports=len(EXPORTS), native=True)))


def verify(root, output, *, native=False):
    output.mkdir(parents=True, exist_ok=False)
    integrity = json.loads((root / "integrity.json").read_bytes())
    for relative, digest in integrity.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    app = None
    if native:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    expected = json.loads((root / "expected.json").read_bytes())
    originals = {
        row["case"]: row for row in json.loads((root / "verification.json").read_bytes())["cases"]
    }
    for case in EXPORTS:
        directory = root / "snapshots" / case
        report = json.loads((directory / "review.json").read_bytes())
        recipe = next(directory.glob("executed_recipe.sbp*"))
        assert hashlib.sha256(recipe.read_bytes()).hexdigest() == report["exported_recipe_sha256"]
        assert not any(
            report[name]
            for name in (
                "planning_reexecuted",
                "clearance_reexecuted",
                "depth_reprojected",
                "recipe_reloaded",
                "training_changed",
                "robot_execution",
            )
        )
        for name, description in report["input_arrays"].items():
            array = np.load(directory / f"input_{name}.npy", allow_pickle=False)
            assert hashlib.sha256(array.tobytes()).hexdigest() == description["sha256"]
    rows = []
    for case, _ in CASES:
        path = root / "projects" / f"{case}.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"], case
            session.run()
            block = picker(session.graph)
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
                    frozen = native_review(app, window, block)
                    assert (
                        frozen.input_description("depth")["sha256"]
                        == originals[case]["source_sha256"]
                    )
                finally:
                    window.dirty = False
                    window.close()
                    app.processEvents()
            target = output / case / "project.json"
            target.parent.mkdir()
            model = block.parameters["recipe_path"]
            raw = (path.parent / model).read_bytes()
            session.save(target)
            assert (target.parent / model).read_bytes() == raw
        with ProjectSession.open(target, registry) as copied:
            copied.run()
            result = picker(copied.graph).last_result
            assert result is not None and stable_result(result) == expected[case]
            assert graphics_hashes(result) == originals[case]["graphics"]
        rows.append(
            dict(
                case=case,
                numeric_exact=True,
                default_clouds_exact=True,
                native_review=native,
                save_as_reopen=True,
            )
        )
    write_json(
        output / "checks.json",
        dict(
            cases=rows,
            exports_verified=len(EXPORTS),
            file_pins_verified=len(integrity),
            scope="Offline parity, not physical qualification",
        ),
    )
    print(
        json.dumps(
            dict(projects=len(rows), exports=len(EXPORTS), pins=len(integrity), native=native)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.mode == "build":
        build(args.source.resolve(), args.output.resolve())
    else:
        verify(args.source.resolve(), args.output.resolve(), native=args.native)
