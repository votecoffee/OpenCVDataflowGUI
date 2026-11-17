"""Native completed-run reference/map reviews from unchanged licensed public recipes."""

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
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin

CASES = (
    ("generated", "", 2),
    ("fixture", "fixture_capture", 3),
    ("visa_limited", "result_review/samples/visa_pcb2_project", 2),
    ("visa_posed", "visa_pcb2", 2),
)
MAPS = ("residual", "deviation_map", "inspected_mask", "defect_mask")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def stable_report(result):
    return dict(
        passed=result.passed,
        failures=list(result.fail_reasons),
        inspection=copy.deepcopy(result.outputs["inspection"]),
        defect_area=result.outputs["defect_area"],
    )


def raster_hashes(result):
    return {
        name: hashlib.sha256(np.asarray(result.outputs[name]).tobytes()).hexdigest()
        for name in (*MAPS, "overlay", "assessment")
        if result.outputs[name] is not NO_VALUE
    }


def load_companion(root, registry):
    load_plugin(root / "pattern_plugin.py", registry)


def ground_truth(root, name, index, source_maps, image, target):
    if name not in ("visa_limited", "visa_posed"):
        return None
    label = "anomaly" if name == "visa_posed" and index == 1 else "normal"
    output = dict(label=label, used_for_inference=False)
    if name == "visa_posed":
        prediction = source_maps["defect_mask"] != 0
        truth = np.zeros(image.shape, bool)
        output["false_alarm"] = label == "normal" and bool(
            np.any(source_maps["defect_mask"])
        )
        if label == "anomaly":
            truth = cv2.imread(str(root / "assets/truth-anomaly-000.png"), 0) != 0
            tp = int(np.count_nonzero(truth & prediction))
            positive = int(np.count_nonzero(truth))
            predicted = int(np.count_nonzero(prediction))
            output.update(
                mask="assets/truth-anomaly-000.png",
                truth_pixels=positive,
                predicted_pixels=predicted,
                true_positive_pixels=tp,
                precision=None if not predicted else tp / predicted,
                recall=None if not positive else tp / positive,
            )
        # Example-only evaluation graphic. Labels/masks are never graph inputs.
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        views = [base.copy() for _ in range(3)]
        views[0][truth] = (35, 180, 245)
        views[1][prediction] = (65, 65, 245)
        views[2][truth & prediction] = (80, 225, 95)
        views[2][prediction & ~truth] = (65, 65, 245)
        views[2][truth & ~prediction] = (35, 180, 245)
        h, w = image.shape
        panel = np.full((h + 84, w * 3, 3), 22, np.uint8)
        for position, title in enumerate(
            (
                "Dataset label: " + label,
                "Accepted prediction",
                "Green TP | Red false positive | Amber missed",
            )
        ):
            left = position * w
            panel[50 : 50 + h, left : left + w] = views[position]
            cv2.putText(
                panel,
                title,
                (left + 8, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        precision = output.get("precision")
        recall = output.get("recall")
        precision_text = "n/a" if precision is None else f"{precision:.1%}"
        recall_text = "n/a" if recall is None else f"{recall:.1%}"
        verdict = (
            "Normal image false alarm."
            if output["false_alarm"]
            else "Normal image, no accepted change."
        )
        text = "Source pixels; evaluation only, never used for training/inference. " + (
            verdict
            if label == "normal"
            else f"Precision {precision_text}; recall {recall_text}."
        )
        cv2.putText(
            panel,
            text,
            (8, h + 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".png", panel)
        assert ok
        (target / "ground_truth_comparison.png").write_bytes(encoded.tobytes())
        output["comparison"] = "ground_truth_comparison.png"
    return output


def build(source, output):
    from PySide6.QtWidgets import QApplication
    from opencv_dataflow.app import MainWindow

    output.mkdir(parents=True, exist_ok=False)
    parent = Path(__file__).resolve().parent
    for name in (
        ".gitattributes",
        "plugin.py",
        "pattern_plugin.py",
        "LICENSE",
        "COMPLETED_REVIEW.md",
        "wheel-verification.json",
        "pattern-wheel-verification.json",
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
    registry = builtin_registry()
    load_companion(output, registry)
    load_plugin(output / "plugin.py", registry)
    from opencv_simple_reference_inspect import dataflow
    from opencv_simple_reference_inspect.model import ReferenceModel

    app = QApplication.instance() or QApplication([])
    expected, receipt = {}, []
    for name, original, frames in CASES:
        folder = output / name
        folder.mkdir()
        shutil.copytree(source / original / "assets", folder / "assets")
        shutil.copy2(source / original / "project.json", folder / "project.json")
        if name == "visa_posed":
            for document in ("SOURCES.md", "source_manifest.json"):
                shutil.copy2(source / original / document, folder / document)
            shutil.copy2(
                source / original / "evaluation.json",
                folder / "upstream-evaluation.json",
            )
        with ProjectSession.open(folder / "project.json", registry) as session:
            if name == "visa_posed":
                # First held-out normal and first held-out anomaly, by original
                # publisher-split order; never select using a prediction score.
                session.graph.blocks["source"].parameters["paths"] = [
                    "assets/test-normal-0011.jpg",
                    "assets/test-anomaly-000.jpg",
                ]
                session.save(folder / "project.json")
            assert session.inspect()["validation"]["valid"]
            window = MainWindow()
            warnings = []
            window.warn = warnings.append
            try:
                load_companion(output, window.registry)
                load_plugin(output / "plugin.py", window.registry)
                window.replace_graph(session.graph, folder / "project.json")
                window.show()
                for index in range(frames):
                    session.run()
                    block = session.graph.blocks["inspection"]
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
                            "load_model",
                            side_effect=AssertionError("Reloaded model"),
                        ),
                        patch.object(
                            ReferenceModel,
                            "inspect",
                            side_effect=AssertionError("Inspected again"),
                        ),
                        patch(
                            "opencv_simple_reference_inspect.model._normalize",
                            side_effect=AssertionError("Refit exposure"),
                        ),
                    ):
                        inspector.open_plugin_review_dialog()
                        app.processEvents()
                        popup = inspector._plugin_review_dialog
                        assert popup is not None and popup.snapshot is not None, (
                            warnings
                        )
                        capture, panel = popup.snapshot, popup.panel
                        panel._actual_size()
                        scale = panel.right.transform().m11()
                        for mode in range(panel.mode.count()):
                            panel.mode.setCurrentIndex(mode)
                            assert panel.right.transform().m11() == scale
                            assert not panel.right.image_item.pixmap().isNull()
                        panel.tabs.setCurrentIndex(1)
                        panel.tabs.setCurrentIndex(2)
                        assert not panel.sheet.image_item.pixmap().isNull()
                        assert not popup.source_view.image_item.pixmap().isNull()
                        target = output / "snapshots" / f"{name}_{index}"
                        panel._export_handler(target)
                        assert stable_report(block.last_result) == before
                        assert raster_hashes(block.last_result) == graphics
                        assert block.parameters == parameters
                        summary = panel.review.summary()
                        receipt.append(
                            dict(
                                case=name,
                                frame=index,
                                state=summary["state"],
                                original_recipe_sha256=capture.record.recipe_sha256,
                                source_sha256=capture.source_sha256,
                                source_shape=list(capture.source_pixels.shape),
                                source_dtype=capture.source_pixels.dtype.str,
                                source_format=capture.source_format,
                                reference_size=list(capture.model.mean.shape[::-1]),
                                runtime_rasters=graphics,
                                native_results_action=True,
                                source_and_comparisons_visible=True,
                                review_did_not_inspect_refit_or_reload=True,
                                primary_values_and_rasters_unchanged=True,
                                parameters_unchanged=True,
                                export=target.relative_to(output).as_posix(),
                                ground_truth=ground_truth(
                                    folder,
                                    name,
                                    index,
                                    capture.source_maps,
                                    capture.image,
                                    target,
                                ),
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
                exported_reviews=len(receipt),
                states=[r["state"] for r in receipt],
                bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Existing public reference_inspect root"
    )
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
