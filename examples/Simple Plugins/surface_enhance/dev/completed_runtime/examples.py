"""Build/replay frozen surface reviews with fixed settings and labeled rendered inputs."""

import argparse
import copy
import hashlib
import json
import shutil
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin

CASES = (
    ("analytic_clean", "analytic_clean"),
    ("analytic_contaminated", "analytic_contaminated"),
    ("four_light_ambiguity", "four_light_ambiguity"),
    ("material_clean", "polyhaven_clean"),
    ("material_contaminated", "polyhaven_contaminated"),
    ("connected_material", "polyhaven_contaminated"),
    ("exposure_changed", "analytic_clean"),
    ("missing", "analytic_clean"),
    ("graphics_off", "analytic_contaminated"),
)
EXPORTS = {
    "analytic_clean",
    "four_light_ambiguity",
    "material_contaminated",
    "connected_material",
    "exposure_changed",
    "missing",
}


def write_json(path, value):
    path.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode() + b"\n")


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
                    if name in ("report", "coverage", "valid_pixels")
                },
            ),
            allow_nan=False,
        )
    )


def array_hashes(result):
    def array(value):
        return dict(
            shape=list(value.shape),
            dtype=value.dtype.str,
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )

    return dict(
        maps={name: array(value) for name, value in result.outputs["maps"].items()},
        images={
            name: array(value)
            for name, value in result.outputs.items()
            if isinstance(value, np.ndarray)
        },
        graphics={name: array(value) for name, value in result.graphics.items()},
    )


def no_analysis():
    from opencv_simple_surface_enhance.solver import SurfaceEstimator

    stack = ExitStack()
    stack.enter_context(
        patch.object(SurfaceEstimator, "estimate", side_effect=AssertionError("Fit"))
    )
    for name in (
        "dataflow.execute",
        "dataflow.import_capture",
        "dataflow.load_recipe",
        "execution.execute",
        "execution.prepare_capture",
        "execution.prepare_stack",
        "recipe.import_capture",
        "recipe.load_recipe",
        "render.relight",
        "render.normal_change",
        "render.render_views",
    ):
        stack.enter_context(
            patch("opencv_simple_surface_enhance." + name, side_effect=AssertionError(name))
        )
    return stack


def native_review(app, window, block, target=None):
    before, arrays = stable_result(block.last_result), array_hashes(block.last_result)
    parameters = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    with (
        no_analysis(),
        patch.object(block, "create_plugin_dialog", side_effect=AssertionError("Opened training")),
    ):
        inspector.open_plugin_review_dialog()
        app.processEvents()
        popup = inspector._plugin_review_dialog
        assert popup.snapshot is not None and popup.selected_pixel is None
        frozen = popup.snapshot
        if block.last_result.outputs["overview"] is not NO_VALUE:
            np.testing.assert_array_equal(
                frozen.view("overview")[0], block.last_result.outputs["overview"]
            )
        reasons = frozen.record._output["maps"]["reason"]
        rows = np.argwhere((reasons != 0) & (reasons != 1))
        if not len(rows):
            rows = np.argwhere(reasons == 0)
        y, x = (int(v) for v in (rows[len(rows) // 2] if len(rows) else [0, 0]))
        popup.source_view.pixel_chosen.emit(x, y)
        popup.focus_pixel()
        zoom = popup.source_view.transform().m11()
        popup.map_view.pixel_chosen.emit(x, y)
        assert popup.source_view.transform().m11() == zoom
        assert popup.table.rowCount() == frozen.light_count
        popup.light.setCurrentIndex(min(1, frozen.light_count - 1))
        for name in ("observation_flags", "used", "light_residual", "quality"):
            popup.map.setCurrentIndex(popup.map.findData(name))
        if target is not None:
            popup.map.setCurrentIndex(popup.map.findData("normals"))
            popup.export_to(target)
        popup.source.setCurrentIndex(popup.source.findData("reference"))
        assert "Training reference" in popup.view_note.text()
        popup.source.setCurrentIndex(0)
        popup.overview()
        assert popup.selected_pixel is None and not popup.inspect.isChecked()
        assert stable_result(block.last_result) == before
        assert array_hashes(block.last_result) == arrays and block.parameters == parameters
        popup.reject()
        assert popup.snapshot is None and popup.source_view.image_item.pixmap().isNull()
        app.processEvents()
    return frozen


def asset(projects, raw, suffix):
    relative = f"assets/{hashlib.sha256(raw).hexdigest()}{suffix}"
    target = projects / relative
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        assert target.read_bytes() == raw
    else:
        target.write_bytes(raw)
    return relative


def manifest_asset(projects, path):
    data = json.loads(path.read_bytes())
    for row in data["lights"]:
        original = path.parent / row["image"]
        row["image"] = Path(asset(projects, original.read_bytes(), original.suffix)).name
    if data.get("mask"):
        original = path.parent / data["mask"]
        data["mask"] = Path(asset(projects, original.read_bytes(), original.suffix)).name
    return asset(projects, json.dumps(data, indent=2, allow_nan=False).encode() + b"\n", ".json")


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
    shutil.copytree(source / "studies/polyhaven_source", output / "sources")
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    from opencv_simple_surface_enhance import LightRig
    from opencv_simple_surface_enhance.recipe import import_capture, save_recipe

    app = QApplication.instance() or QApplication([])
    projects = output / "projects"
    projects.mkdir()
    expected, records = {}, []
    for case, base in CASES:
        original = source / "execution" / f"{base}.json"
        path = projects / f"{case}.json"
        shutil.copy2(original, path)
        with ProjectSession.open(path, registry) as session:
            block = session.graph.blocks["surface"]
            old_model = original.parent / block.parameters["recipe_path"]
            block.parameters["recipe_path"] = asset(projects, old_model.read_bytes(), ".sse")
            original_capture = original.parent / block.parameters["capture_manifest"]
            if case in ("connected_material", "exposure_changed", "missing", "graphics_off"):
                capture = import_capture(original_capture)
                if case == "exposure_changed":
                    rig = LightRig(
                        capture.rig.light_ids,
                        capture.rig.directions,
                        capture.rig.relative_intensity,
                        capture.rig.exposure_s * 0.5,
                        capture.rig.frame,
                    )
                    capture = replace(
                        capture,
                        rig=rig,
                        capture_id="analytic-half-exposure",
                        reference=capture.black_level
                        + (capture.reference - capture.black_level) * 0.5,
                    )
                elif case == "missing":
                    capture = replace(
                        capture,
                        capture_id="analytic-all-missing",
                        reference=np.zeros_like(capture.reference),
                    )
                temporary = output.parent / f"{output.name}-{case}.sse"
                save_recipe(temporary, capture)
                raw = temporary.read_bytes()
                node = registry.create(
                    "input.simple_surface_capture",
                    block_id="capture",
                    parameters=dict(
                        capture_path=asset(projects, raw, ".sse"),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    ),
                )
                session.graph.add(node)
                node.position = (0, 0)
                block.position = (470, 0)
                session.graph.connect(node.id, "capture", block.id, "capture")
                block.parameters.update(source="connected capture", capture_manifest="")
            else:
                block.parameters["capture_manifest"] = manifest_asset(projects, original_capture)
            if case == "graphics_off":
                block.parameters["render_graphics"] = False
            block.description = (
                "Review the exact completed lights/maps/pixel evidence in the "
                "inspector. Rendered control; no acquired-camera qualification."
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
                records.append(
                    dict(
                        case=case,
                        passed=block.last_result.passed,
                        valid=report["valid_pixels"],
                        active=report["active_pixels"],
                        source_mode=block.parameters["source"],
                        capture_id=report["capture_id"],
                        coverage=report["coverage"],
                        reasons=report["reason_counts"],
                        recipe_sha256=hashlib.sha256(frozen.recipe_bytes).hexdigest(),
                        capture_sha256=hashlib.sha256(
                            frozen.record._capture["images"].tobytes()
                        ).hexdigest(),
                        arrays=array_hashes(block.last_result),
                        native_results=True,
                        no_preparation_estimation_relight_or_reload=True,
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
            cases=records,
            projects=len(records),
            frozen_exports=len(EXPORTS),
            scope="Rendered software controls, not acquired-camera qualification",
        ),
    )
    for path in output.rglob("*"):
        if path.is_file() and (
            path.suffix in (".py", ".md", ".json") or path.name in ("LICENSE", ".gitattributes")
        ):
            raw = path.read_bytes()
            assert not raw.startswith(b"\xef\xbb\xbf")
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
    print(json.dumps(dict(projects=len(records), exports=len(EXPORTS), native=True)))


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
        folder = root / "snapshots" / case
        doc = json.loads((folder / "review.json").read_bytes())
        assert (
            hashlib.sha256((folder / "executed_recipe.sse").read_bytes()).hexdigest()
            == doc["exported_recipe_sha256"]
        )
        assert not any(
            doc[key]
            for key in (
                "preparation_reexecuted",
                "estimation_reexecuted",
                "calibration_changed",
                "capture_reloaded",
                "recipe_reloaded",
                "training_changed",
            )
        )
        for key in ("images", "mask"):
            record = doc["consumed_capture"][key]
            if record is not None:
                array = np.load(folder / f"capture_{key}.npy", allow_pickle=False)
                assert hashlib.sha256(array.tobytes()).hexdigest() == record["sha256"]
    rows = []
    for case, _ in CASES:
        path = root / "projects" / f"{case}.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"], case
            session.run()
            block = session.graph.blocks["surface"]
            assert block.last_result is not None, block.error
            assert stable_result(block.last_result) == expected[case], case
            assert array_hashes(block.last_result) == originals[case]["arrays"], case
            if app is not None:
                from opencv_dataflow.app import MainWindow

                window = MainWindow()
                load_plugin(root / "plugin.py", window.registry)
                window.replace_graph(session.graph, path)
                window.show()
                try:
                    frozen = native_review(app, window, block)
                    assert (
                        hashlib.sha256(frozen.record._capture["images"].tobytes()).hexdigest()
                        == originals[case]["capture_sha256"]
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
            result = copied.graph.blocks["surface"].last_result
            assert result is not None and stable_result(result) == expected[case]
            assert array_hashes(result) == originals[case]["arrays"]
        rows.append(
            dict(
                case=case,
                numeric_maps_and_graphics_exact=True,
                native_review=native,
                exact_model_save_as_reopen=True,
            )
        )
    write_json(
        output / "checks.json",
        dict(
            cases=rows,
            exports_verified=len(EXPORTS),
            file_pins_verified=len(pins),
            scope="Software parity, not physical calibration/material truth",
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
