"""Build/replay exact completed fringe reviews using fixed generic settings."""

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

CASES = (
    "analytic_clean",
    "analytic_missing",
    "gray_phase_conflict",
    "geometry_rejected",
    "connected_missing",
    "missing",
    "graphics_off",
    "warsaw_crown_phase",
)
EXPORTS = {
    "analytic_clean",
    "gray_phase_conflict",
    "geometry_rejected",
    "connected_missing",
    "missing",
    "warsaw_crown_phase",
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
                    if name in ("report", "coverage", "count", "metric")
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

    cloud = result.outputs["cloud"]
    return dict(
        maps={name: array(value) for name, value in result.outputs["maps"].items()},
        images={
            name: array(value)
            for name, value in result.outputs.items()
            if isinstance(value, np.ndarray)
        },
        graphics={name: array(value) for name, value in result.graphics.items()},
        cloud=None
        if cloud is NO_VALUE
        else {
            name: None if getattr(cloud, name) is None else array(getattr(cloud, name))
            for name in ("points", "pixels", "colors")
        },
    )


def no_analysis():
    stack = ExitStack()
    for name in (
        "dataflow.execute",
        "dataflow.import_capture",
        "dataflow.load_recipe",
        "execution.execute",
        "cli.run",
        "cli.read_manifest",
        "cli.load_plane",
        "recipe.import_capture",
        "recipe.load_recipe",
        "decode.decode",
        "cli.decode",
        "phase.demodulate",
        "decode.demodulate",
        "cli.demodulate",
        "geometry.triangulate",
        "cli.triangulate",
        "render.views",
        "cli.views",
    ):
        stack.enter_context(
            patch("opencv_simple_fringe_scan." + name, side_effect=AssertionError(name))
        )
    return stack


def native_review(app, window, block, target=None):
    before, arrays = stable_result(block.last_result), array_hashes(block.last_result)
    parameters = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    with (
        no_analysis(),
        patch.object(block, "create_plugin_dialog", side_effect=AssertionError("Training")),
    ):
        inspector.open_plugin_review_dialog()
        app.processEvents()
        popup = inspector._plugin_review_dialog
        assert popup.snapshot is not None, popup.status.text()
        frozen = popup.snapshot
        if block.last_result.outputs["overview"] is not NO_VALUE:
            np.testing.assert_array_equal(
                frozen.view("overview")[0], block.last_result.outputs["overview"]
            )
        maps = frozen.record._output["maps"]
        reasons = maps.get("geometry_reason", maps["reason"])
        rows = np.argwhere(reasons != 0)
        if not len(rows):
            rows = np.argwhere(reasons == 0)
        y, x = (int(v) for v in rows[len(rows) // 2])
        popup.source_view.pixel_chosen.emit(x, y)
        popup.focus_pixel()
        zoom = popup.source_view.transform().m11()
        popup.map_view.pixel_chosen.emit(x, y)
        assert popup.source_view.transform().m11() == zoom
        assert popup.table.rowCount() == frozen.frame_count
        popup.frame.setCurrentIndex(1 if "wrapped" in maps else frozen.frame_count - 12)
        for name in ("reason", "relative_rms", "wrapped" if "wrapped" in maps else "wrapped_phase"):
            popup.map.setCurrentIndex(popup.map.findData(name))
        if frozen.cloud is not None:
            row = min(5, len(frozen.cloud.points) - 1)
            popup.cloud_picker.point_picked.emit(
                dict(row=row, source_identity=popup.cloud_picker.source.identity)
            )
            assert popup.selected_pixel == tuple(int(v) for v in frozen.cloud.pixels[row])
            popup.rig.setChecked(True)
            popup.tabs.setCurrentIndex(popup.native_index)
            app.processEvents()
            popup.cloud_view.flyaround.setChecked(True)
            popup.cloud_view.advance_orbit()
            popup.cloud_view.flyaround.setChecked(False)
            popup.select_pixel(x, y)
        else:
            assert popup.cloud_view.cloud is None and not popup.tabs.isTabEnabled(
                popup.native_index
            )
        popup.map.setCurrentIndex(
            popup.map.findData("geometry_reason" if frozen.record.report["metric"] else "reason")
        )
        if target is not None:
            name = (
                "geometry_ray_gap_m"
                if "geometry_rejected" in str(target)
                else "gray_phase_error_px"
                if "gray_phase_conflict" in str(target)
                else "geometry_xyz_m"
                if "analytic_clean" in str(target)
                else "wrapped"
                if "wrapped" in maps
                else "wrapped_phase"
            )
            popup.map.setCurrentIndex(popup.map.findData(name))
            if name == "geometry_xyz_m":
                popup.component.setCurrentIndex(2)
            popup.export_to(target)
        popup.source.setCurrentIndex(1)
        assert "training reference" in popup.view_note.text()
        popup.source.setCurrentIndex(0)
        popup.overview()
        assert popup.selected_pixel is None
        assert stable_result(block.last_result) == before
        assert array_hashes(block.last_result) == arrays and block.parameters == parameters
        popup.reject()
        assert popup.snapshot is None and popup.cloud_view.cloud is None
        assert not popup.cloud_view.orbit_timer.isActive()
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
    for row in data["frames"]:
        original = path.parent / row["image"]
        row["image"] = Path(asset(projects, original.read_bytes(), original.suffix)).name
    if data.get("mask"):
        original = path.parent / data["mask"]
        data["mask"] = Path(asset(projects, original.read_bytes(), original.suffix)).name
    return asset(projects, json.dumps(data, indent=2, allow_nan=False).encode() + b"\n", ".json")


def altered_capture(recipe, case):
    """Independent analytic pattern controls: alter observations, never fit gates."""
    from opencv_simple_fringe_scan.recipe import FringeRecipe

    data = recipe.definition
    data["capture_id"] = case
    raw = recipe.reference.copy()
    if case == "missing":
        raw[:] = 0
    else:
        yy, xx = np.indices(raw.shape[1:])
        xn, yn = (xx - 48) / 120, (yy - 36) / 120
        depth = 0.75 / (1 + 0.12 * xn + 0.05 * yn)
        u, v = 120 * (xn * depth - 0.08) / depth + 64, 120 * yn + 48
        region = np.zeros(raw.shape[1:], bool)
        region[15:57, 30:65] = True
        angles = 2 * np.pi * np.arange(6) / 6
        if case == "gray_phase_conflict":
            # X phase moves 2px while Gray remains unchanged: genuine coding disagreement.
            for i, angle in enumerate(angles):
                raw[30 + i, region] = (0.45 + 0.32 * np.cos(2 * np.pi * (u + 2) / 16 + angle))[
                    region
                ]
        else:
            # Coherent Y Gray+phase shift: code accepted, stereo rays become inconsistent.
            v = v + region * 2
            integer = np.floor(v + 0.5).astype(np.uint16)
            gray = integer ^ (integer >> 1)
            index = 16
            for bit in range(6, -1, -1):
                signal = ((gray >> bit) & 1) * 0.74
                raw[index, region] = (0.08 + signal)[region]
                raw[index + 1, region] = (0.82 - signal)[region]
                index += 2
            for i, angle in enumerate(angles):
                raw[36 + i, region] = (0.45 + 0.32 * np.cos(2 * np.pi * v / 16 + angle))[region]
    return FringeRecipe(data, raw, recipe.mask)


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
        "compare.py",
    ):
        shutil.copy2(template / name, output / name)
    shutil.copytree(template / "releases", output / "releases")
    shutil.copytree(source / "studies/warsaw_source", output / "sources")
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    from opencv_simple_fringe_scan.recipe import import_capture, save_recipe

    app = QApplication.instance() or QApplication([])
    projects = output / "projects"
    projects.mkdir()
    expected, records = {}, []
    for case in CASES:
        phase = case == "warsaw_crown_phase"
        training = import_capture(
            source / "studies" / (case if phase else "analytic_clean") / "capture.json"
        )
        training_path = output.parent / f"{output.name}-training-{case}.sfs"
        save_recipe(training_path, training)
        model = training_path.read_bytes()
        path = projects / f"{case}.json"
        with ProjectSession(registry, directory=projects) as session:
            session.apply(
                [
                    dict(
                        op="add",
                        type="3d.simple_fringe_scan",
                        id="fringe",
                        position=[450, 0],
                        parameters=dict(
                            recipe_path=asset(projects, model, ".sfs"),
                            sha256=hashlib.sha256(model).hexdigest(),
                            minimum_coverage=0.95,
                        ),
                    )
                ]
            )
            block = session.graph.blocks["fringe"]
            if case in ("gray_phase_conflict", "geometry_rejected", "missing", "connected_missing"):
                capture = (
                    import_capture(source / "studies/analytic_missing/capture.json")
                    if case == "connected_missing"
                    else altered_capture(training, case)
                )
                temporary = output.parent / f"{output.name}-capture-{case}.sfs"
                save_recipe(temporary, capture)
                raw = temporary.read_bytes()
                session.apply(
                    [
                        dict(
                            op="add",
                            type="input.simple_fringe_capture",
                            id="capture",
                            position=[0, 0],
                            parameters=dict(
                                capture_path=asset(projects, raw, ".sfs"),
                                sha256=hashlib.sha256(raw).hexdigest(),
                            ),
                        ),
                        dict(
                            op="connect",
                            source="capture",
                            output="capture",
                            target="fringe",
                            input="capture",
                        ),
                    ]
                )
            else:
                base = "analytic_missing" if case == "graphics_off" else case
                block.parameters.update(
                    source="capture manifest",
                    capture_manifest=manifest_asset(
                        projects, source / "studies" / base / "capture.json"
                    ),
                )
            block = session.graph.blocks["fringe"]
            if case == "graphics_off":
                block.parameters["render_graphics"] = False
            block.node_size, block.thumbnail = "large", True
            block.preview_output = "overview" if phase else "cloud"
            block.description = (
                "Exact completed pattern/map/cloud review. Simulated controls; "
                "no physical accuracy qualification."
            )
            session.save(path)
            assert session.inspect()["validation"]["valid"]
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
                        metric=report["metric"],
                        reasons=report["reason_counts"],
                        geometry_reasons=report.get("geometry", {}).get("reason_counts"),
                        recipe_sha256=hashlib.sha256(frozen.recipe_bytes).hexdigest(),
                        capture_sha256=hashlib.sha256(
                            frozen.record._capture["images"].tobytes()
                        ).hexdigest(),
                        arrays=array_hashes(block.last_result),
                        native_results=True,
                        no_decoding_unwrapping_triangulation_or_reload=True,
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
            scope="Simulated controls, not acquired-camera or physical accuracy",
        ),
    )
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
            hashlib.sha256((folder / "executed_recipe.sfs").read_bytes()).hexdigest()
            == doc["exported_recipe_sha256"]
        )
        assert not any(
            doc[key]
            for key in (
                "decoding_reexecuted",
                "unwrapping_reexecuted",
                "triangulation_reexecuted",
                "calibration_changed",
                "capture_reloaded",
                "recipe_reloaded",
                "training_changed",
            )
        )
        for key, declaration in json.loads((folder / "capture_data.json").read_bytes()).items():
            if isinstance(declaration, dict) and "array_file" in declaration:
                array = np.load(folder / declaration["array_file"], allow_pickle=False)
                assert hashlib.sha256(array.tobytes()).hexdigest() == declaration["sha256"], key
    rows = []
    for case in CASES:
        path = root / "projects" / f"{case}.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.run()
            block = session.graph.blocks["fringe"]
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
            result = copied.graph.blocks["fringe"].last_result
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
            scope="Software parity, not physical calibration/shape/coding truth",
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
