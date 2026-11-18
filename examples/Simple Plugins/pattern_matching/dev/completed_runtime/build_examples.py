"""Copy licensed prepared examples and export actual completed runs through native Results review."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin

CASES = (
    ("brackets", "asymmetric_brackets", "project.json", 2),
    ("wired", "asymmetric_brackets", "wired_recipe.json", 2),
    ("visa_pcb2", "visa_pcb2", "project.json", 2),
    ("blank", "asymmetric_brackets", "project.json", 1),
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def stable_report(result):
    return dict(
        count=result.outputs["count"],
        score=result.outputs["score"],
        model_id=result.outputs["matches"]["pattern_id"],
        items=[
            dict(
                score=item["score"],
                coverage=item["coverage"],
                fit_error_working_px=item["fit_error_working_px"],
                coarse_score=item["coarse_score"],
                contrast=item["contrast"],
                clutter=item["clutter"],
                outside_region_fraction=item["outside_region_fraction"],
                measured_edge_count=item["measured_edge_count"],
                source_to_input=item["pose"]["source_to_target"],
            )
            for item in result.outputs["matches"]["items"]
        ],
        rejection_counts=result.outputs["review"]["rejection_counts"],
        working_size=result.outputs["review"]["working_size"],
        working_to_input=result.outputs["review"]["working_to_input"],
    )


def build(source, output):
    from PySide6.QtWidgets import QApplication
    from opencv_dataflow.app import MainWindow

    output.mkdir(parents=True, exist_ok=False)
    parent = Path(__file__).resolve().parent
    for name in (
        "plugin.py",
        "LICENSE",
        "COMPLETED_REVIEW.md",
        "wheel-verification.json",
        "README.md",
        "SOURCES.md",
        "build_examples.py",
        "verify_examples.py",
        "baseline-comparison.json",
        "native-replay.json",
        "host-native-replay.json",
    ):
        shutil.copy2(parent / name, output / name)
    shutil.copytree(parent / "releases", output / "releases")
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    from opencv_geometric_pattern_match import dataflow

    app = QApplication.instance() or QApplication([])
    receipt, expected = [], {}
    for name, original, project_name, frames in CASES:
        folder = output / name
        folder.mkdir()
        shutil.copytree(source / original / "assets", folder / "assets")
        shutil.copy2(source / original / project_name, folder / "project.json")
        with ProjectSession.open(folder / "project.json", registry) as session:
            assert session.inspect()["validation"]["valid"]
            if name == "blank":
                original_image = cv2.imread(str(folder / "assets/tray.png"), 0)
                assert cv2.imwrite(
                    str(folder / "assets/blank.png"), np.full_like(original_image, 40)
                )
                session.graph.blocks["scene"].parameters["paths"] = ["assets/blank.png"]
                session.save(folder / "project.json")
            window = MainWindow()
            warnings = []
            window.warn = warnings.append
            try:
                load_plugin(output / "plugin.py", window.registry)
                window.replace_graph(session.graph, folder / "project.json")
                window.show()
                for index in range(frames):
                    session.run()
                    block = session.graph.blocks["matcher"]
                    assert block.last_result is not None, block.error
                    before = stable_report(block.last_result)
                    expected[f"{name}:{index}"] = copy.deepcopy(before)
                    overlay = np.asarray(block.last_result.outputs["overlay"]).copy()
                    before_parameters = copy.deepcopy(block.parameters)
                    inspector = window.inspector
                    inspector.select(block)
                    with (
                        patch.object(
                            dataflow,
                            "review_search",
                            side_effect=AssertionError("Review searched"),
                        ),
                        patch.object(
                            dataflow,
                            "locate_all",
                            side_effect=AssertionError("Review searched"),
                        ),
                        patch.object(
                            dataflow._PinnedRecipeBlock,
                            "load_pinned_model",
                            side_effect=AssertionError("Review reloaded model"),
                        ),
                    ):
                        inspector.open_plugin_review_dialog()
                        app.processEvents()
                        popup = inspector._plugin_review_dialog
                        assert popup is not None and popup.snapshot is not None, (
                            warnings,
                            popup.status.text() if popup else "",
                        )
                        capture = popup.snapshot
                        assert capture.record.model.model_id == before["model_id"]
                        assert len(capture.record.matches) == before["count"]
                        assert (
                            not popup.fit_view.image.isNull()
                            and not popup.source_view.image.isNull()
                        )
                        destinations = []
                        if capture.record.matches:
                            choice = min(
                                capture.record.matches,
                                key=lambda candidate: candidate.score,
                            )
                            row = next(
                                i
                                for i, candidate in enumerate(popup._visible)
                                if candidate is choice
                            )
                            popup.table.selectRow(row)
                            popup.focus()
                            path = output / "snapshots" / f"{name}_{index}_accepted"
                            popup.export_to(path)
                            destinations.append(path.relative_to(output).as_posix())
                        else:
                            popup.overview()
                            path = output / "snapshots" / f"{name}_{index}_no_match"
                            popup.export_to(path)
                            destinations.append(path.relative_to(output).as_posix())
                        score_rejections = [
                            c
                            for c in capture.record.rejected
                            if c.reason == "score_cutoff"
                        ]
                        if score_rejections:
                            popup.kind.setCurrentIndex(1)
                            choice = score_rejections[0]
                            row = next(
                                i for i, c in enumerate(popup._visible) if c is choice
                            )
                            popup.table.selectRow(row)
                            path = output / "snapshots" / f"{name}_{index}_rejected"
                            popup.export_to(path)
                            destinations.append(path.relative_to(output).as_posix())
                        for control in popup.layers.values():
                            control.setChecked(False)
                            control.setChecked(True)
                        assert before == stable_report(block.last_result)
                        np.testing.assert_array_equal(
                            overlay, np.asarray(block.last_result.outputs["overlay"])
                        )
                        assert block.parameters == before_parameters
                        receipt.append(
                            dict(
                                case=name,
                                frame=index,
                                accepted=before["count"],
                                model_sha256=capture.record.model.sha256,
                                source_sha256=capture.source_sha256,
                                source_size=list(capture.source_pixels.shape[1::-1]),
                                source_dtype=capture.source_pixels.dtype.str,
                                working_size=list(capture.record.working_size),
                                native_readonly_results_action=True,
                                source_and_fit_visible=True,
                                review_did_not_search=True,
                                primary_values_and_overlay_unchanged=True,
                                parameters_unchanged=True,
                                runtime_overlay_sha256=hashlib.sha256(
                                    overlay.tobytes()
                                ).hexdigest(),
                                exports=destinations,
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
                cases=len(CASES),
                frames=len(receipt),
                exported_reviews=sum(len(r["exports"]) for r in receipt),
                accepted=[r["accepted"] for r in receipt],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Existing public pattern_matching examples root"
    )
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
