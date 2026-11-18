"""Rebuild native completed-run reviews from existing public PathWidth recipes."""

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
    ("generated", "", 2),
    ("calibrated", "calibrated_bead", 3),
    ("weld", "weld_bead", 3),
)
GROUPS = ("out_of_tolerance", "missing", "ambiguous", "unobserved")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def matcher(graph):
    return next(
        b for b in graph.blocks.values() if b.type_id == "inspect.simple_path_width"
    )


def stable_report(result):
    names = (
        "measurements",
        "mean_width",
        "coverage",
        "longest_gap",
        "physical_measurements",
        "mean_width_physical",
        "longest_gap_physical",
    )
    return json.loads(
        json.dumps(
            dict(
                passed=result.passed,
                score=result.score,
                failures=list(result.fail_reasons),
                outputs={
                    name: None
                    if result.outputs[name] is NO_VALUE
                    else result.outputs[name]
                    for name in names
                },
            ),
            allow_nan=False,
        )
    )


def raster_hashes(result):
    return {
        name: hashlib.sha256(np.asarray(result.outputs[name]).tobytes()).hexdigest()
        for name in ("overlay", "trend", "physical_trend")
        if result.outputs[name] is not NO_VALUE
    }


def representative(indices):
    runs = np.split(np.asarray(indices), np.where(np.diff(indices) != 1)[0] + 1)
    run = max(runs, key=len)
    return int(run[len(run) // 2])


def build(source, output):
    from PySide6.QtWidgets import QApplication
    from opencv_dataflow.app import MainWindow

    output.mkdir(parents=True, exist_ok=False)
    parent = Path(__file__).resolve().parent
    for name in (
        ".gitattributes",
        "plugin.py",
        "LICENSE",
        "COMPLETED_REVIEW.md",
        "wheel-verification.json",
        "README.md",
        "SOURCES.md",
        "build_examples.py",
        "verify_examples.py",
    ):
        shutil.copy2(parent / name, output / name)
    for name in (
        "baseline-comparison.json",
        "native-replay.json",
        "host-native-replay.json",
    ):
        if (parent / name).is_file():
            shutil.copy2(parent / name, output / name)
    shutil.copytree(parent / "releases", output / "releases")
    shutil.copy2(
        source / "weld_bead/LICENSE-data.txt", output / "WELD-LICENSE-data.txt"
    )
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    from opencv_simple_path_width import dataflow
    from opencv_simple_path_width.recipe import PathWidthRecipe
    from opencv_simple_path_width.station_review import station_indices

    app = QApplication.instance() or QApplication([])
    expected, receipt = {}, []
    for name, original, frames in CASES:
        folder = output / name
        folder.mkdir()
        shutil.copytree(source / original / "assets", folder / "assets")
        shutil.copy2(source / original / "project.json", folder / "project.json")
        if original == "weld_bead":
            for document in ("SOURCES.md", "LICENSE-data.txt"):
                shutil.copy2(source / original / document, folder / document)
        with ProjectSession.open(folder / "project.json", registry) as session:
            assert session.inspect()["validation"]["valid"]
            window = MainWindow()
            warnings = []
            window.warn = warnings.append
            try:
                load_plugin(output / "plugin.py", window.registry)
                window.replace_graph(session.graph, folder / "project.json")
                window.show()
                for index in range(frames):
                    session.run()
                    block = matcher(session.graph)
                    assert block.last_result is not None, block.error
                    before = stable_report(block.last_result)
                    graphics = raster_hashes(block.last_result)
                    parameters = copy.deepcopy(block.parameters)
                    expected[f"{name}:{index}"] = before
                    inspector = window.inspector
                    inspector.select(block)
                    with (
                        patch.object(
                            dataflow,
                            "_read_pinned",
                            side_effect=AssertionError("Reloaded"),
                        ),
                        patch.object(
                            PathWidthRecipe,
                            "measure",
                            side_effect=AssertionError("Measured again"),
                        ),
                        patch(
                            "opencv_simple_path_width.measurement._edge_pair",
                            side_effect=AssertionError("Paired again"),
                        ),
                        patch(
                            "opencv_simple_path_width.measurement._continuous_pair",
                            side_effect=AssertionError("Paired again"),
                        ),
                    ):
                        inspector.open_plugin_review_dialog()
                        app.processEvents()
                        popup = inspector._plugin_review_dialog
                        assert popup is not None and popup.snapshot is not None, (
                            warnings
                        )
                        capture = popup.snapshot
                        result = capture.record.result
                        assert not popup.source_view.image_item.pixmap().isNull()
                        assert not popup.panel.image.image_item.pixmap().isNull()
                        choices = [
                            (group, station_indices(result, group)) for group in GROUPS
                        ]
                        choices = [
                            (group, indices) for group, indices in choices if indices
                        ]
                        if not choices:
                            choices = [("measured", station_indices(result, "all"))]
                        exports = []
                        for group, indices in choices:
                            selected = representative(indices)
                            panel = popup.panel
                            panel._image_selected(*result.stations[selected].point)
                            assert panel.selected_station == selected
                            trace = panel.trace
                            panel._focus_station()
                            panel.select_station(
                                result.stations[selected].distance_px
                                / result.path_length_px
                            )
                            assert panel.trace is trace
                            if result.physical_measurements is not None:
                                panel.tabs.setCurrentIndex(0)
                                panel.tabs.setCurrentIndex(1)
                            path = output / "snapshots" / f"{name}_{index}_{group}"
                            panel._export_handler(path, selected)
                            assert panel.trace is trace
                            exports.append(
                                dict(
                                    path=path.relative_to(output).as_posix(),
                                    group=group,
                                    station=selected,
                                )
                            )
                        assert stable_report(block.last_result) == before
                        assert raster_hashes(block.last_result) == graphics
                        assert block.parameters == parameters
                        receipt.append(
                            dict(
                                case=name,
                                frame=index,
                                passed=result.passed,
                                stations=len(result.stations),
                                coverage=result.coverage,
                                manifest_sha256=capture.record.manifest_sha256,
                                source_sha256=capture.source_sha256,
                                source_shape=list(capture.source_pixels.shape),
                                source_dtype=capture.source_pixels.dtype.str,
                                source_format=capture.source_format,
                                groups=capture.report()["station_groups"],
                                runtime_rasters=graphics,
                                native_results_action=True,
                                source_and_measurement_visible=True,
                                review_did_not_measure_or_reload=True,
                                primary_values_and_rasters_unchanged=True,
                                parameters_unchanged=True,
                                exports=exports,
                            )
                        )
                        popup.reject()
                        app.processEvents()
                    assert not warnings, warnings
                session.save(folder / "project.json")
            finally:
                window.dirty = False
                window.close()
                app.processEvents()
    write_json(output / "expected.json", expected)
    write_json(output / "native-verification.json", receipt)
    print(
        json.dumps(
            dict(
                projects=len(CASES),
                frames=len(receipt),
                exported_reviews=sum(len(r["exports"]) for r in receipt),
                passed=[r["passed"] for r in receipt],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Existing public path_width example root"
    )
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
