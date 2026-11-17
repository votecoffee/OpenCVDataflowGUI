"""Native completed segmentation review using licensed real images and unchanged models."""

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
    ("tote", "armbench_tote", 2),
    ("pcb", "pcb_laminate", 2),
    ("votes_unavailable", "pcb_laminate", 1),
    ("heatmap_only", "armbench_tote", 2),
    ("unobserved", "armbench_tote", 1),
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def stable_report(result):
    return json.loads(
        json.dumps(
            dict(
                passed=result.passed,
                count=result.outputs["count"],
                report=result.outputs["report"],
            ),
            allow_nan=False,
        )
    )


def raster_hashes(result):
    return {
        name: hashlib.sha256(np.asarray(result.outputs[name]).tobytes()).hexdigest()
        for name in ("mask", "inspected", "score_map", "overlay", "heatmap")
        if result.outputs[name] is not NO_VALUE
    }


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
        "ARMBENCH-License-CC-BY.txt",
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
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    from opencv_defect_segmentation import PixelSegmenter

    app = QApplication.instance() or QApplication([])
    expected, receipt = {}, []
    for name, original, frames in CASES:
        folder = output / name
        shutil.copytree(
            source / original,
            folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        with ProjectSession.open(folder / "project.json", registry) as session:
            block = session.graph.blocks["segment"]
            if name in ("votes_unavailable", "unobserved"):
                session.graph.blocks["source"].parameters["paths"] = (
                    session.graph.blocks["source"].parameters["paths"][:1]
                )
            if name == "votes_unavailable":
                block.parameters.update(emit_score_map=False, render_heatmap=False)
            if name == "heatmap_only":
                block.parameters.update(emit_score_map=False, render_heatmap=True)
            if name == "unobserved":
                control = session.graph.add(
                    registry.create(
                        "cv.threshold",
                        block_id="validity_control",
                        parameters={"value": 0.0, "max_value": 0.0},
                    )
                )
                control.name = "Control: deliberately no observed pixels"
                control.position = (560.0, 440.0)
                session.graph.connect("blue", "image", control.id, "image")
                session.graph.connect(control.id, "image", "segment", "valid")
            session.save(folder / "project.json")
            assert session.inspect()["validation"]["valid"], session.inspect()[
                "validation"
            ]
            window = MainWindow()
            warnings = []
            window.warn = warnings.append
            try:
                load_plugin(output / "plugin.py", window.registry)
                window.replace_graph(session.graph, folder / "project.json")
                window.show()
                for index in range(frames):
                    session.run()
                    assert block.last_result is not None, block.error
                    before = stable_report(block.last_result)
                    graphics = raster_hashes(block.last_result)
                    parameters = copy.deepcopy(block.parameters)
                    expected[f"{name}:{index}"] = before
                    inspector = window.inspector
                    inspector.select(block)
                    with (
                        patch.object(
                            block,
                            "_verified_assets",
                            side_effect=AssertionError("Loaded model"),
                        ),
                        patch.object(
                            PixelSegmenter,
                            "segment",
                            side_effect=AssertionError("Inferred again"),
                        ),
                        patch.object(
                            PixelSegmenter,
                            "_from_yaml",
                            side_effect=AssertionError("Decoded model"),
                        ),
                    ):
                        inspector.open_plugin_review_dialog()
                        app.processEvents()
                        popup = inspector._plugin_review_dialog
                        assert popup is not None and popup.snapshot is not None, (
                            warnings
                        )
                        capture = popup.snapshot
                        popup.part.setCurrentIndex(1)
                        regions = capture.record.report["parts"][0]["regions"]
                        if regions:
                            selected = max(
                                range(len(regions)), key=lambda i: regions[i]["area_px"]
                            )
                            popup.table.selectRow(selected)
                            assert popup.selected_region == selected
                        popup.focus()
                        for tab in range(popup.tabs.count()):
                            popup.tabs.setCurrentIndex(tab)
                            app.processEvents()
                            assert (
                                not popup.tabs.currentWidget()
                                .image_item.pixmap()
                                .isNull()
                            )
                        target = output / "snapshots" / f"{name}_{index}"
                        popup.export_to(target)
                        assert stable_report(block.last_result) == before
                        assert raster_hashes(block.last_result) == graphics
                        assert block.parameters == parameters
                        summary = capture.report(
                            popup.selected_part, popup.selected_region
                        )
                        receipt.append(
                            dict(
                                case=name,
                                frame=index,
                                state=summary["state"],
                                source_sha256=capture.source_sha256,
                                source_shape=list(capture.source_pixels.shape),
                                source_dtype=capture.source_pixels.dtype.str,
                                source_format=capture.source_format,
                                model_sha256=capture.record.report["model_sha256"],
                                region_sha256=capture.record.report["region_sha256"],
                                score_source=capture.record.score_source,
                                retained_score_sha256=None
                                if capture.scores is None
                                else hashlib.sha256(
                                    capture.scores.tobytes()
                                ).hexdigest(),
                                source_foreground_pixels=summary[
                                    "source_foreground_pixels"
                                ],
                                source_inspected_pixels=summary[
                                    "source_inspected_pixels"
                                ],
                                selected_part=popup.selected_part,
                                selected_region=popup.selected_region,
                                runtime_rasters=graphics,
                                native_results_action=True,
                                all_review_tabs_visible=True,
                                review_did_not_infer_decode_or_reload=True,
                                primary_values_and_rasters_unchanged=True,
                                parameters_unchanged=True,
                                export=target.relative_to(output).as_posix(),
                            )
                        )
                        popup.reject()
                        app.processEvents()
                    assert not warnings, warnings
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
                exports=len(receipt),
                states=[r["state"] for r in receipt],
                foreground=[r["source_foreground_pixels"] for r in receipt],
                bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Existing public pixel_segmentation root"
    )
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
