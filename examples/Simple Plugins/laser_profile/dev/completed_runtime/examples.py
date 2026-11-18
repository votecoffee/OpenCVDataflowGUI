"""Build/replay exact completed laser-profile reviews using unchanged public recipes."""

import argparse
import copy
import hashlib
import json
import shutil
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import image_format
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import PointCloud

CASES = (
    ("calibrated_step", "calibrated_step"),
    ("calibrated_unavailable", "calibrated_unavailable"),
    ("calibrated_missing", "calibrated_step"),
    ("geometry_rejected", "calibrated_step"),
    ("graphics_off", "calibrated_step"),
    ("real_wli1", "wli_1"),
    ("real_wli2", "wli_2"),
    ("real_wli3", "wli_3"),
    ("real_wli4", "wli_4"),
)
EXPORTS = {
    "calibrated_step",
    "calibrated_unavailable",
    "calibrated_missing",
    "geometry_rejected",
    "real_wli1",
    "real_wli3",
}


def write_json(path, value):
    path.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n")


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
                    if value is not NO_VALUE and not isinstance(value, (np.ndarray, PointCloud))
                },
            ),
            allow_nan=False,
        )
    )


def graphics_hashes(result):
    values = {}
    for name, value in result.outputs.items():
        if isinstance(value, np.ndarray):
            values[name] = dict(
                dtype=value.dtype.str,
                shape=list(value.shape),
                encoding=image_format(value),
                sha256=hashlib.sha256(value.tobytes()).hexdigest(),
            )
        elif isinstance(value, PointCloud):
            values[name] = dict(
                units=value.units,
                frame=value.frame,
                arrays={
                    key: None
                    if getattr(value, key) is None
                    else hashlib.sha256(getattr(value, key).tobytes()).hexdigest()
                    for key in ("points", "pixels", "colors", "normals")
                },
            )
    return values


def no_analysis():
    from opencv_simple_laser_profile import dataflow

    stack = ExitStack()
    for name in ("execute", "load_recipe", "source_image"):
        stack.enter_context(patch.object(dataflow, name, side_effect=AssertionError(name)))
    for name in (
        "profile.extract_stripe",
        "profile.intersect_pixels",
        "stripe.extract_stripe",
        "geometry.intersect_pixels",
        "recipe.load_recipe",
        "recipe.teach",
        "geometry.cv2.undistortPoints",
    ):
        stack.enter_context(
            patch("opencv_simple_laser_profile." + name, side_effect=AssertionError(name))
        )
    return stack


def native_review(app, window, block, target=None):
    before, images = stable_result(block.last_result), graphics_hashes(block.last_result)
    parameters = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    with (
        no_analysis(),
        patch.object(block, "create_plugin_dialog", side_effect=AssertionError("Opened teaching")),
    ):
        inspector.open_plugin_review_dialog()
        app.processEvents()
        popup = inspector._plugin_review_dialog
        assert popup.snapshot is not None and popup.selected_bin is None
        frozen = popup.snapshot
        if block.last_result.outputs["overlay"] is not NO_VALUE:
            np.testing.assert_array_equal(frozen.overlay(), block.last_result.outputs["overlay"])
            np.testing.assert_array_equal(frozen.chart(), block.last_result.outputs["chart"])
        cloud = block.last_result.outputs["cloud"]
        if cloud is not NO_VALUE:
            np.testing.assert_array_equal(popup.cloud_view.cloud.points, cloud.points)
            np.testing.assert_array_equal(popup.cloud_view.cloud.colors, cloud.colors)
            popup.result_tabs.setCurrentIndex(popup.native_index)
            app.processEvents()
            popup.cloud_view.flyaround.setChecked(True)
            assert popup.cloud_view.orbit_timer.isActive()
            yaw = popup.cloud_view.yaw.value()
            popup.cloud_view.advance_orbit()
            assert popup.cloud_view.yaw.value() != yaw
        else:
            assert not popup.result_tabs.isTabEnabled(popup.native_index)
        rows = frozen.record.report["samples"]
        row = next(
            (r for r in rows if not r["valid"] and r["reason"] != "outside_roi"),
            next((r for r in rows if r["valid"]), rows[0]),
        )
        popup.overlay_view.bin_chosen.emit(row["bin"])
        assert popup.selected_bin == row["bin"]
        popup.focus_sample()
        zoom = popup.overlay_view.transform().m11()
        popup.chart_view.bin_chosen.emit(row["bin"])
        assert popup.overlay_view.transform().m11() == zoom
        if cloud is not NO_VALUE:
            popup.layer_boxes["laser_plane"].setChecked(True)
            popup.layer_boxes["ray_segment"].setChecked(True)
        if target is not None:
            popup.export_to(target)
        popup.overview()
        if cloud is not NO_VALUE:
            np.testing.assert_array_equal(popup.cloud_view.cloud.points, cloud.points)
        assert stable_result(block.last_result) == before
        assert graphics_hashes(block.last_result) == images and block.parameters == parameters
        popup.reject()
        assert popup.snapshot is None and popup.cloud_view.cloud is None
        assert popup.cloud_view._rendered is None and not popup.cloud_view.orbit_timer.isActive()
        app.processEvents()
    return frozen


def asset(projects, raw, suffix):
    relative = f"assets/{hashlib.sha256(raw).hexdigest()}{suffix}"
    destination = projects / relative
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        assert destination.read_bytes() == raw
    else:
        destination.write_bytes(raw)
    return relative


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
    for name in ("provenance.json", "summary.json"):
        shutil.copy2(source / "studies/wli" / name, sources / name)
    shutil.copytree(source / "studies/wli/centerline", sources / "centerline")
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    from opencv_simple_laser_profile import Plane, load_recipe, save_recipe

    app = QApplication.instance() or QApplication([])
    projects = output / "projects"
    projects.mkdir()
    expected, rows = {}, []
    for case, base in CASES:
        original = source / "execution" / base
        path = projects / f"{case}.json"
        shutil.copy2(original / "project.json", path)
        with ProjectSession.open(path, registry) as session:
            for block in session.graph.blocks.values():
                names = (
                    ("path",) if block.type_id == "input.file" else block.project_asset_parameters
                )
                for name in names:
                    current = original / block.parameters[name]
                    block.parameters[name] = asset(projects, current.read_bytes(), current.suffix)
            block = session.graph.blocks["laser"]
            if case == "graphics_off":
                block.parameters["render_graphics"] = False
            elif case == "calibrated_missing":
                source_block = session.graph.blocks["source"]
                image = cv2.imread(
                    str(projects / source_block.parameters["path"]), cv2.IMREAD_UNCHANGED
                )
                okay, raw = cv2.imencode(".png", np.zeros_like(image))
                assert okay
                source_block.parameters["path"] = asset(projects, raw.tobytes(), ".png")
            elif case == "geometry_rejected":
                recipe = load_recipe(
                    projects / block.parameters["recipe_path"], sha256=block.parameters["sha256"]
                )
                temporary = output.parent / f"{output.name}-behind-camera.slp"
                save_recipe(temporary, replace(recipe, laser=Plane([0, 0, 1], 1)))
                raw = temporary.read_bytes()
                block.parameters.update(
                    recipe_path=asset(projects, raw, ".slp"), sha256=hashlib.sha256(raw).hexdigest()
                )
            block.description = (
                "Independent completed Results: source/trend/bin evidence and "
                "native calibrated cloud. Availability is not gauge accuracy."
            )
            session.save(path)
            assert session.inspect()["validation"]["valid"], case
            session.run()
            assert block.last_result is not None, block.error
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
                        valid=report["valid_samples"],
                        active=report["active_sample_count"],
                        calibrated=report["calibrated"],
                        reasons=report["reason_counts"],
                        recipe_sha256=hashlib.sha256(frozen.recipe_bytes).hexdigest(),
                        source_sha256=frozen.source_description["sha256"],
                        graphics=graphics_hashes(block.last_result),
                        native_results=True,
                        no_extraction_intersection_calibration_or_reload=True,
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
            scope="Offline software evidence, not gauge qualification",
        ),
    )
    for path in output.rglob("*"):
        if path.is_file() and (
            path.suffix in (".py", ".md", ".json") or path.name in ("LICENSE", ".gitattributes")
        ):
            raw = path.read_bytes()
            assert not raw.startswith(b"\xef\xbb\xbf"), path
            path.write_bytes(raw.replace(b"\r\n", b"\n"))
    write_json(
        output / "integrity.json",
        {
            str(path.relative_to(output)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        },
    )
    print(json.dumps(dict(projects=len(rows), exports=len(EXPORTS), native=True)))


def verify(root, output, *, native=False):
    output.mkdir(parents=True, exist_ok=False)
    pins = json.loads((root / "integrity.json").read_bytes())
    for relative, digest in pins.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    app = None
    if native:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    expected = json.loads((root / "expected.json").read_bytes())
    originals = {
        r["case"]: r for r in json.loads((root / "verification.json").read_bytes())["cases"]
    }
    for case in EXPORTS:
        directory = root / "snapshots" / case
        report = json.loads((directory / "review.json").read_bytes())
        assert (
            hashlib.sha256((directory / "executed_recipe.slp").read_bytes()).hexdigest()
            == report["exported_recipe_sha256"]
        )
        source = np.load(directory / "source_image.npy", allow_pickle=False)
        assert hashlib.sha256(source.tobytes()).hexdigest() == report["source"]["sha256"]
        assert not any(
            report[name]
            for name in (
                "extraction_reexecuted",
                "intersection_reexecuted",
                "calibration_changed",
                "recipe_reloaded",
                "training_changed",
            )
        )
        if not report["emitted_cloud"]:
            assert not list(directory.glob("*.ply"))
    rows = []
    for case, _ in CASES:
        path = root / "projects" / f"{case}.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"], case
            session.run()
            block = session.graph.blocks["laser"]
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
                    assert frozen.source_description["sha256"] == originals[case]["source_sha256"]
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
            result = copied.graph.blocks["laser"].last_result
            assert result is not None and stable_result(result) == expected[case]
            assert graphics_hashes(result) == originals[case]["graphics"]
        rows.append(
            dict(
                case=case,
                numeric_exact=True,
                primary_and_graphics_exact=True,
                native_review=native,
                save_as_reopen=True,
            )
        )
    write_json(
        output / "checks.json",
        dict(
            cases=rows,
            exports_verified=len(EXPORTS),
            file_pins_verified=len(pins),
            scope="Software parity, not physical gauge qualification",
        ),
    )
    print(json.dumps(dict(projects=len(rows), exports=len(EXPORTS), pins=len(pins), native=native)))


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
