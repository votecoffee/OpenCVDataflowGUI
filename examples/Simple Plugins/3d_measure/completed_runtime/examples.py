"""Build/replay native completed measurement reviews from unchanged public recipes."""

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
    ("real13", "residuals/tless_frame13"),
    ("real24", "residuals/tless_frame24"),
    ("good", "residuals/good"),
    ("raised", "residuals/raised"),
    ("missing", "residuals/missing"),
    ("line_guard", "residuals/line_nonparallel"),
    ("flatness", "flatness/generated/good"),
    ("graphics_off", "residuals/tless_frame13"),
)
EXPORTS = {"real13", "raised", "missing", "line_guard", "flatness"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_result(result):
    return json.loads(
        json.dumps(
            dict(
                passed=result.passed,
                outputs={
                    name: value
                    for name, value in result.outputs.items()
                    if name
                    in (
                        "measurements",
                        "planes",
                        "lines",
                        "report",
                        "value",
                        "count",
                        "residual_report",
                    )
                    and value is not NO_VALUE
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
        for name in ("fit", "residuals")
        if (value := result.outputs[name]) is not NO_VALUE
    }


def native_review(app, window, block, target=None):
    from opencv_simple_3d_measure import Measurer, dataflow

    before, charts = stable_result(block.last_result), graphics_hashes(block.last_result)
    parameters = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    with (
        patch.object(Measurer, "measure", side_effect=AssertionError("Measured")),
        patch.object(dataflow, "load_recipe", side_effect=AssertionError("Loaded recipe")),
        patch.object(block, "create_plugin_dialog", side_effect=AssertionError("Opened training")),
        patch(
            "opencv_simple_3d_measure.measure.fit_plane", side_effect=AssertionError("Fit plane")
        ),
        patch("opencv_simple_3d_measure.measure.fit_line", side_effect=AssertionError("Fit line")),
        patch(
            "opencv_simple_3d_measure.measure.minimum_zone_flatness",
            side_effect=AssertionError("Fit zone"),
        ),
    ):
        inspector.open_plugin_review_dialog()
        app.processEvents()
        popup = inspector._plugin_review_dialog
        assert popup is not None and popup.snapshot is not None
        capture = popup.snapshot
        if "fit" in charts:
            np.testing.assert_array_equal(
                popup.cloud.points, block.last_result.outputs["fit"].points
            )
        popup.flyaround.setChecked(True)
        assert popup.orbit_timer.isActive()
        yaw = popup.yaw.value()
        popup.advance_orbit()
        assert popup.yaw.value() != yaw
        popup.flyaround.setChecked(False)
        measurements = capture.record.report["measurements"]
        selected = next(
            (m for m in measurements if m["status"] in ("out_of_limits", "unavailable")),
            measurements[0],
        )
        popup.select_measurement(selected["name"])
        region = selected["other"] or selected["plane"]
        popup.region.setCurrentIndex(popup.region.findData(region))
        popup.mode.setCurrentIndex(popup.mode.findData("residuals"))
        popup.focus_region()
        assert popup.histogram.row is not None
        cache = capture.residual_review()
        popup.layers["axes"].setChecked(False)
        assert capture.residual_review() is cache
        if target is not None:
            popup.export_to(target)
        popup.mode.setCurrentIndex(popup.mode.findData("input"))
        np.testing.assert_array_equal(popup.cloud.points, capture.input_cloud().points)
        assert (
            stable_result(block.last_result) == before
            and graphics_hashes(block.last_result) == charts
        )
        assert block.parameters == parameters
        popup.reject()
        app.processEvents()
    return capture, selected["name"], region


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
    shutil.copy2(source / "real_tless/ATTRIBUTION.md", sources / "ATTRIBUTION.md")
    shutil.copy2(source / "real_tless/source/provenance.json", sources / "provenance.json")
    projects = output / "projects"
    projects.mkdir()
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    app = QApplication.instance() or QApplication([])
    expected, rows, content_paths = {}, [], {}
    for case, relative in CASES:
        original = source / relative
        path = projects / f"{case}.json"
        shutil.copy2(original / "project.json", path)
        with ProjectSession.open(path, registry) as session:
            for block in session.graph.blocks.values():
                names = (
                    ("recipe_path",)
                    if block.type_id == "3d.simple_measure"
                    else ("path",)
                    if block.type_id == "input.cloud"
                    else ()
                )
                for name in names:
                    current = (original / block.parameters[name]).resolve()
                    if current not in content_paths:
                        raw = current.read_bytes()
                        digest = hashlib.sha256(raw).hexdigest()
                        relative_asset = f"assets/{digest}{current.suffix}"
                        target = projects / relative_asset
                        target.parent.mkdir(exist_ok=True)
                        if target.exists():
                            assert target.read_bytes() == raw
                        else:
                            target.write_bytes(raw)
                        content_paths[current] = relative_asset
                    block.parameters[name] = content_paths[current]
            block = session.graph.blocks["measure"]
            if case == "graphics_off":
                block.parameters.update(render_graphics=False, residual_graphics=False)
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
                target = output / "snapshots" / case if case in EXPORTS else None
                capture, selected, region = native_review(app, window, block, target)
                rows.append(
                    dict(
                        case=case,
                        passed=block.last_result.passed,
                        count=block.last_result.outputs["count"],
                        source_sha256=hashlib.sha256(
                            capture.input_cloud().points.tobytes()
                        ).hexdigest(),
                        recipe_sha256=hashlib.sha256(capture.recipe_bytes).hexdigest(),
                        graphics=graphics_hashes(block.last_result),
                        selected_measurement=selected,
                        selected_region=region,
                        native_results_flyaround=True,
                        no_refit_measurement_reload_or_training=True,
                        primary_outputs_and_graphics_unchanged=True,
                        export=None if target is None else target.relative_to(output).as_posix(),
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
                exports=sum(r["export"] is not None for r in rows),
                passed=[r["passed"] for r in rows],
                counts=[r["count"] for r in rows],
            )
        )
    )


def verify(root, output, *, native=False, core_only=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    if core_only:
        from opencv_simple_3d_measure.dataflow import register

        registry.types.pop("3d.simple_measure", None)
        register(registry)
    else:
        load_plugin(root / "plugin.py", registry)
    app = None
    if native:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    expected = json.loads((root / "expected.json").read_bytes())
    originals = {r["case"]: r for r in json.loads((root / "native-verification.json").read_bytes())}
    exports = 0
    for row in originals.values():
        if row["export"] is None:
            continue
        target = root / row["export"]
        report = json.loads((target / "review.json").read_bytes())
        assert (
            hashlib.sha256((target / "executed_recipe.s3m").read_bytes()).hexdigest()
            == row["recipe_sha256"]
        )
        assert (
            hashlib.sha256(np.load(target / "consumed_scene_points.npy").tobytes()).hexdigest()
            == row["source_sha256"]
        )
        assert report["completed_result"] == expected[row["case"]]["outputs"]["report"]
        assert not report["fit_reexecuted"] and not report["training_changed"]
        for i, region in enumerate(report["exported_regions"]):
            if not region["available"]:
                assert not (target / f"region_{i}_residual.npy").exists()
        exports += 1
    rows = []
    for case, _ in CASES:
        path = root / "projects" / f"{case}.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.run()
            block = session.graph.blocks["measure"]
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
                    capture, _, _ = native_review(app, window, block)
                    assert (
                        hashlib.sha256(capture.input_cloud().points.tobytes()).hexdigest()
                        == originals[case]["source_sha256"]
                    )
                finally:
                    window.dirty = False
                    window.close()
                    app.processEvents()
            if not core_only:
                target = output / case / "project.json"
                target.parent.mkdir()
                recipe = block.parameters["recipe_path"]
                original = (path.parent / recipe).read_bytes()
                session.save(target)
                assert (target.parent / recipe).read_bytes() == original
        if not core_only:
            with ProjectSession.open(target, registry) as copied:
                copied.run()
                result = copied.graph.blocks["measure"].last_result
                assert result is not None and stable_result(result) == expected[case]
                assert graphics_hashes(result) == originals[case]["graphics"]
        rows.append(
            dict(
                case=case,
                numeric_exact=True,
                default_clouds_exact=True,
                native_review=native,
                save_as_reopen=not core_only,
            )
        )
    write_json(
        output / "checks.json",
        dict(
            core_only=core_only,
            cases=rows,
            exports_verified=exports,
        scope=(
            "Unchanged measurements/fits/residuals and emitted clouds; no accuracy qualification"
        ),
        ),
    )
    print(
        json.dumps(
            dict(
                projects=len(rows),
                exports=exports,
                numeric_exact=True,
                default_clouds_exact=True,
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
