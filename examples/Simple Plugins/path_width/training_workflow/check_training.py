"""Replay the actual native inspector without driving the desktop or installing packages."""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "pinned_path_width_training", ROOT / "plugin.py"
)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.core import Graph
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from opencv_simple_path_width import load_recipe
from opencv_simple_path_width.qt_canvas import MeasurementPreview
from opencv_simple_path_width.station_review import save_station_review


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def draw_planned_guides(canvas, path):
    """Compose the actual foreground renderer onto reference pixels, not a window capture."""
    canvas.actual_size()
    source = canvas.image_item.pixmap()
    image = QImage(source.size(), QImage.Format.Format_RGB32)
    painter = QPainter(image)
    try:
        painter.drawPixmap(0, 0, source)
        canvas.drawForeground(painter, QRectF(image.rect()))
    finally:
        painter.end()
    assert image.save(str(path))


def verify_case(root, name, registry, app):
    folder = root / name
    project = folder / "project.json"
    graph = Graph.load(project, registry)
    block = next(
        b for b in graph.blocks.values() if b.type_id == "inspect.simple_path_width"
    )
    original = copy.deepcopy(block.parameters)
    originals = {
        p.relative_to(folder).as_posix(): digest(p)
        for p in (folder / "assets").rglob("*")
        if p.is_file()
    }
    window = MainWindow()
    warnings = []
    window.warn = warnings.append
    window.registry = registry
    window.replace_graph(graph, project)
    window.show()
    inspector = window.inspector
    inspector.select(block)
    record = {"case": name, "desktop_input_used": False, "window_capture_used": False}
    try:
        with patch(
            "opencv_simple_path_width.recipe.measure_path",
            side_effect=AssertionError("Measured during restore"),
        ):
            inspector.open_plugin_dialog()
            app.processEvents()
            dialog = inspector._plugin_dialog
            assert dialog.reference is not None and dialog._setup is not None
            initial_points = np.asarray(dialog.canvas.vertices).copy()
            dialog.reject()
        assert block.parameters == original
        inspector.open_plugin_dialog()
        dialog = inspector._plugin_dialog
        # Explicit generated affine control: import and align the SAME path, with
        # no application-specific limits, detector tuning or automatic pose search.
        matrix = np.array([[1.25, 0.05, 7], [0.02, 1.3, 11.0]])
        source = (initial_points - matrix[:, 2]) @ np.linalg.inv(matrix[:, :2]).T
        path_file, fit_file = folder / "source_path.json", folder / "landmarks.json"
        write_json(
            path_file,
            {"vertices": source.tolist(), "closed": dialog.closed.isChecked()},
        )
        landmarks = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
        write_json(
            fit_file,
            {
                "source": landmarks.tolist(),
                "reference": (landmarks @ matrix[:, :2].T + matrix[:, 2]).tolist(),
            },
        )
        with patch.object(dialog, "_file", return_value=path_file):
            dialog._open_path()
        with patch.object(dialog, "_file", return_value=fit_file):
            dialog._align()
        assert "RMS" in dialog.status.text()
        np.testing.assert_allclose(dialog.canvas.vertices, initial_points, atol=1e-10)
        dialog.canvas.reverse()
        dialog.canvas.reverse()
        if name == "calibrated":
            with patch.object(
                dialog, "_file", return_value=folder / "assets/calibration.json"
            ):
                dialog._open_calibration()
            assert dialog._setup.length_physical is not None
        dialog.pick_mode.setCurrentIndex(3)
        count = len(dialog._setup.centers)
        dialog.canvas.picked.emit(*dialog._setup.centers[count // 2])
        assert dialog.guide_station.value() == count // 2 + 1
        dialog._focus_guide()
        draw_planned_guides(dialog.canvas, folder / "planned_guides.png")
        record.update(
            planned_stations=count,
            planned_length_px=dialog._setup.length_px,
            planned_length_physical=dialog._setup.length_physical,
            units=dialog._setup.units,
            setup_summary=dialog.setup_summary.text(),
            source_to_reference=dialog.alignment,
        )
        previews = []

        def preview(view):
            view.show()
            app.processEvents()
            np.testing.assert_allclose(
                [s.point for s in view.result.stations], dialog._setup.centers
            )
            issues = [
                i for i, s in enumerate(view.result.stations) if s.status != "pass"
            ]
            index = (
                issues[len(issues) // 2] if issues else len(view.result.stations) // 2
            )
            view.station_number.setValue(index + 1)
            assert view.selected_station == index and view.trace is not None
            view._focus_station()
            save_station_review(
                folder / "preview",
                dialog.reference,
                view.result,
                index,
                settings=view.settings,
                calibration=view.calibration,
            )
            previews.append(view.result)
            view.close()

        with patch.object(MeasurementPreview, "exec", preview):
            dialog._preview()
        assert (
            len(previews) == 1
            and "Current reference preview" in dialog.preview_state.text()
        )
        spacing = dialog.fields["sample_spacing"].text()
        dialog.fields["sample_spacing"].setText("0")
        assert (
            not dialog.stage_button.isEnabled()
            and not dialog.preview_button.isEnabled()
        )
        assert dialog._setup is None and dialog.canvas.setup_guide is None
        assert "stale" in dialog.preview_state.text()
        dialog.fields["sample_spacing"].setText(spacing)
        assert dialog._setup is not None and dialog.stage_button.isEnabled()
        assert block.parameters == original
        dialog._stage()
        assert block.parameters == original
        inspector.apply()
        assert not warnings, warnings
        assert block.parameters["recipe_path"] != original["recipe_path"]
        staged = folder / "staged_project.json"
        graph.save(staged)
        inspector.open_plugin_dialog()
        reopened = inspector._plugin_dialog
        with patch(
            "opencv_simple_path_width.recipe.measure_path",
            side_effect=AssertionError("Measured during setup"),
        ):
            np.testing.assert_allclose(
                reopened.canvas.vertices, initial_points, atol=1e-10
            )
            assert reopened.alignment is not None
            assert (
                reopened.preview_calibration is None
            )  # Preview profile is not silently sealed as runtime calibration.
            reopened.reject()
        target = root / ".local" / name / "relocated/project.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with ProjectSession.open(staged, registry) as session:
            assert session.inspect()["validation"]["valid"]
            member = block.parameters["recipe_path"]
            expected = digest(folder / member)
            session.save(target)
        with ProjectSession.open(target, registry) as relocated:
            assert relocated.inspect()["validation"]["valid"]
            assert digest(target.parent / member) == expected
            saved = load_recipe(target.parent / member)
            np.testing.assert_allclose(
                saved.reference_vertices(), initial_points, atol=1e-10
            )
        for relative, expected_hash in originals.items():
            assert digest(folder / relative) == expected_hash, relative
        record.update(
            restored=True,
            native_import_align_reverse=True,
            guides_do_not_inspect=True,
            linked_station_preview=True,
            invalid_setup_and_stale_preview=True,
            cancel_preserves_applied=True,
            stage_preserves_applied_until_apply=True,
            apply_reopen=True,
            exact_recipe_save_as=True,
            original_assets_preserved=True,
            reference_preview_passed=previews[0].passed,
            reference_preview_coverage=previews[0].coverage,
        )
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
    return record


def verify(root, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("calibrated", "weld"):
        shutil.copytree(
            root / name,
            output / name,
            ignore=shutil.ignore_patterns(
                ".local",
                "__pycache__",
                "preview",
                "planned_guides.png",
                "staged_project.json",
            ),
        )
    registry = make_registry(plugins=[str(ROOT / "plugin.py")])
    app = QApplication.instance() or QApplication([])
    rows = [verify_case(output, name, registry, app) for name in ("calibrated", "weld")]
    write_json(output / "validation_training.json", rows)
    print(
        json.dumps(
            {
                "native_workflows": len(rows),
                "originals_preserved": True,
                "exact_recipe_save_as": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT, args.output.resolve())
