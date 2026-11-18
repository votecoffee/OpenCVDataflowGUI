"""Build/replay exact completed reflected-pattern reviews with fixed generic settings."""

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
    "analytic_change",
    "analytic_missing",
    "connected_change",
    "whole_period_shift",
    "test_mask_restricted",
    "missing",
    "graphics_off",
    "sofast_self_review",
)
EXPORTS = {
    "analytic_clean",
    "analytic_change",
    "analytic_missing",
    "whole_period_shift",
    "missing",
    "sofast_self_review",
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
                    key: value
                    for key, value in result.outputs.items()
                    if key in ("report", "coverage", "count", "changed_pixels")
                    and value is not NO_VALUE
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
        maps={key: array(value) for key, value in result.outputs["maps"].items()},
        images={
            key: array(value)
            for key, value in result.outputs.items()
            if isinstance(value, np.ndarray)
        },
        graphics={key: array(value) for key, value in result.graphics.items()},
    )


def no_analysis():
    from opencv_simple_deflectometry.core import CompiledReference
    from opencv_simple_deflectometry.geometry import ReflectionModel

    stack = ExitStack()
    for typ, method in (
        (CompiledReference, "fit"),
        (CompiledReference, "compare"),
        (ReflectionModel, "estimate"),
    ):
        stack.enter_context(patch.object(typ, method, side_effect=AssertionError(method)))
    for name in (
        "dataflow.execute",
        "dataflow.read_capture",
        "dataflow.load_recipe",
        "execution.execute",
        "execution.retain_bundle",
        "cli.read_capture",
        "recipe.import_reference",
        "recipe.load_recipe",
        "core.demodulate",
        "render.comparison_views",
    ):
        stack.enter_context(
            patch("opencv_simple_deflectometry." + name, side_effect=AssertionError(name))
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
        rows = np.argwhere(maps["changed"])
        if not len(rows):
            rows = np.argwhere(maps["reason"] > 1)
        if not len(rows):
            rows = np.argwhere(maps["valid"])
        y, x = (int(v) for v in rows[len(rows) // 2])
        popup.source_view.pixel_chosen.emit(x, y)
        popup.focus_pixel()
        scale = popup.source_view.transform().m11()
        popup.map_view.pixel_chosen.emit(x, y)
        assert popup.source_view.transform().m11() == scale
        assert popup.table.rowCount() == frozen.frame_count
        popup.frame.setCurrentIndex(min(1, frozen.frame_count - 1))
        popup.component.setCurrentIndex(1)
        for key in (
            "test_reason",
            "reference_wrapped",
            "test_relative_rms",
            "near_wrap_boundary",
            "changed",
        ):
            popup.map.setCurrentIndex(popup.map.findData(key))
        if target is not None:
            popup.map.setCurrentIndex(popup.map.findData("overlay"))
            popup.export_to(target)
        popup.source.setCurrentIndex(1)
        assert "training reference" in popup.view_note.text()
        popup.source.setCurrentIndex(0)
        popup.overview()
        assert popup.selected_pixel is None
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
    for row in data["frames"]:
        original = path.parent / row["image"]
        row["image"] = Path(asset(projects, original.read_bytes(), original.suffix)).name
    if data.get("mask"):
        original = path.parent / data["mask"]
        data["mask"] = Path(asset(projects, original.read_bytes(), original.suffix)).name
    return asset(projects, json.dumps(data, indent=2, allow_nan=False).encode() + b"\n", ".json")


def altered_capture(recipe, case):
    from opencv_simple_deflectometry.core import Capture
    from opencv_simple_deflectometry.recipe import DeflectRecipe

    ref = recipe.reference
    x, y, mask = ref.x_images.copy(), ref.y_images.copy(), ref.mask.copy()
    if case == "missing":
        x[:] = 0
        y[:] = 0
    elif case == "test_mask_restricted":
        mask[10:35, 20:60] = False
    else:
        # An independent analytic 2pi phase change leaves every sampled cosine
        # unchanged; it demonstrates the unavoidable whole-period ambiguity.
        yy, xx = np.indices(mask.shape)
        phase = (2 * np.pi * xx / 32 + 0.3, 2 * np.pi * yy / 32 + 0.5)
        for axis, output in zip((ref.sequence.x, ref.sequence.y), (x, y)):
            p = phase[0 if output is x else 1]
            for i, shift in enumerate(axis.phase_angles):
                output[i] = 0.4 + 0.2 * np.cos(p + shift + 2 * np.pi)
    capture = Capture(ref.sequence, case, x, y, mask)
    return DeflectRecipe(
        capture,
        notes=dict(
            source_provenance={
                "kind": "Analytic phase/mask control; no acquired or metric shape truth",
                "whole_period_shift_rad": 2 * np.pi if case == "whole_period_shift" else None,
            }
        ),
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
        "license-verification.json",
        "examples.py",
        "compare.py",
    ):
        shutil.copy2(template / name, output / name)
    shutil.copytree(template / "releases", output / "releases")
    shutil.copytree(source / "sofast_source", output / "sources")
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    from opencv_simple_deflectometry.recipe import import_reference, save_recipe

    app = QApplication.instance() or QApplication([])
    projects = output / "projects"
    projects.mkdir()
    trained = {}
    for key, original in (
        ("analytic", source / "studies/reference/capture.json"),
        ("real", source / "sofast_source/capture.json"),
    ):
        recipe = import_reference(original, maximum_change_rad=0.45)
        path = output.parent / f"{output.name}-training-{key}.sdef"
        save_recipe(path, recipe)
        trained[key] = recipe, path.read_bytes()
    expected, records = {}, []
    for case in CASES:
        training, raw = trained["real" if case == "sofast_self_review" else "analytic"]
        path = projects / f"{case}.json"
        with ProjectSession(registry, directory=projects) as session:
            session.apply(
                [
                    dict(
                        op="add",
                        type="image.simple_deflectometry",
                        id="deflect",
                        position=[450, 0],
                        parameters=dict(
                            recipe_path=asset(projects, raw, ".sdef"),
                            sha256=hashlib.sha256(raw).hexdigest(),
                        ),
                    )
                ]
            )
            block = session.graph.blocks["deflect"]
            if case in (
                "connected_change",
                "whole_period_shift",
                "test_mask_restricted",
                "missing",
            ):
                capture = (
                    import_reference(source / "studies/analytic_change/capture.json")
                    if case == "connected_change"
                    else altered_capture(training, case)
                )
                temporary = output.parent / f"{output.name}-capture-{case}.sdef"
                save_recipe(temporary, capture)
                body = temporary.read_bytes()
                node = registry.create(
                    "input.simple_deflect_capture",
                    block_id="capture",
                    parameters=dict(
                        capture_path=asset(projects, body, ".sdef"),
                        sha256=hashlib.sha256(body).hexdigest(),
                    ),
                )
                node.position = (0, 0)
                session.graph.add(node)
                session.graph.connect(node.id, "capture", block.id, "capture")
            else:
                original = (
                    source / "sofast_source/capture.json"
                    if case == "sofast_self_review"
                    else source
                    / "studies"
                    / ("analytic_missing" if case == "graphics_off" else case)
                    / "capture.json"
                )
                block.parameters.update(
                    source="capture manifest", capture_manifest=manifest_asset(projects, original)
                )
            if case == "graphics_off":
                block.parameters["render_graphics"] = False
            block.node_size, block.thumbnail, block.preview_output = "large", True, "overlay"
            block.description = (
                "Exact completed reference/inspection pattern evidence. "
                "Wrapped phase only; no measured shape."
            )
            session.save(path)
            assert session.inspect()["validation"]["valid"]
            session.run()
            assert block.last_result is not None, block.error
            assert block.last_result.passed == (
                case in ("analytic_clean", "whole_period_shift", "sofast_self_review")
            )
            if case == "whole_period_shift":
                assert block.last_result.outputs["report"]["maximum_observed_change_rad"] < 1e-6
            if case == "test_mask_restricted":
                assert (
                    block.last_result.outputs["count"] == int(training.reference.mask.sum()) - 1000
                )
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
                        comparable=report["comparable_pixels"],
                        active=report["active_pixels"],
                        source_mode=block.parameters["source"],
                        capture_id=report["capture_id"],
                        coverage=report["coverage"],
                        changed=report["changed_pixels"],
                        reasons=report["comparison_reason_counts"],
                        response=report["response"],
                        exposure_s=report["exposure_s"],
                        recipe_sha256=hashlib.sha256(frozen.recipe_bytes).hexdigest(),
                        capture_sha256={
                            axis: hashlib.sha256(
                                frozen.record._capture[axis + "_images"].tobytes()
                            ).hexdigest()
                            for axis in ("x", "y")
                        },
                        arrays=array_hashes(block.last_result),
                        native_results=True,
                        no_phase_fitting_unwrapping_reflection_or_reload=True,
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
            scope="Real self-review and analytic controls; no held-out or physical accuracy claim",
        ),
    )
    write_json(
        output / "integrity.json",
        {
            p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
    )
    print(json.dumps(dict(projects=len(records), exports=len(EXPORTS), native=True)))


def verify(root, output, *, native=False):
    output.mkdir(parents=True, exist_ok=False)
    pins = json.loads((root / "integrity.json").read_bytes())
    for name, digest in pins.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
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
            hashlib.sha256((folder / "executed_recipe.sdef").read_bytes()).hexdigest()
            == doc["exported_recipe_sha256"]
        )
        assert not any(
            doc[key]
            for key in (
                "phase_fitting_reexecuted",
                "unwrapping_reexecuted",
                "reflection_solving_reexecuted",
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
        assert not list(folder.glob("*.ply"))
    rows = []
    for case in CASES:
        path = root / "projects" / f"{case}.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.run()
            block = session.graph.blocks["deflect"]
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
                    assert {
                        axis: hashlib.sha256(
                            frozen.record._capture[axis + "_images"].tobytes()
                        ).hexdigest()
                        for axis in ("x", "y")
                    } == originals[case]["capture_sha256"]
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
            result = copied.graph.blocks["deflect"].last_result
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
            scope="Software parity, not held-out inspection, physical calibration or shape truth",
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
