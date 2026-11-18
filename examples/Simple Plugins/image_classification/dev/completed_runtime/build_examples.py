"""Build native completed classification reviews from unchanged licensed recipes."""

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

CASES = (("surface_texture", 16), ("color_and_layout", 16), ("no_overlay", 1))
EXPORTS = (0, 4, 8, 12)
PORTS = ("label", "score", "confusion", "probabilities")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_result(result):
    return dict(passed=result.passed, **{name: result.outputs[name] for name in PORTS})


def raster_hash(result):
    image = result.outputs["overlay"]
    return None if image is NO_VALUE else hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def build(source, output):
    from opencv_dataflow.app import MainWindow
    from PySide6.QtWidgets import QApplication

    output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).resolve().parent
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
        shutil.copy2(template / name, output / name)
    shutil.copytree(template / "releases", output / "releases")
    folder = output / "vegqual"
    folder.mkdir()
    for name in ("assets", "images"):
        shutil.copytree(source / name, folder / name)
    for name in ("SOURCES.md", "ground_truth.json"):
        shutil.copy2(source / name, folder / name)
    shutil.copy2(source / "project.json", folder / "surface_texture.json")
    truth = [
        row
        for row in json.loads((source / "ground_truth.json").read_bytes())
        if row["source_split"] == "test"
    ]
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    from opencv_simple_classifier import dataflow
    from opencv_simple_classifier.model import Classifier

    with ProjectSession.open(folder / "surface_texture.json", registry) as session:
        session.graph.blocks["classifier"].name = "SimpleClassify — Surface texture"
        session.save(folder / "surface_texture.json")
        block = session.graph.blocks["classifier"]
        block.parameters["model"] = "assets/image_classification/vegqual-v4/manifest.json"
        block.name = "SimpleClassify — Color and layout"
        session.save(folder / "color_and_layout.json")
    with ProjectSession.open(folder / "surface_texture.json", registry) as session:
        session.graph.blocks["test_images"].parameters["paths"] = [truth[0]["image"]]
        block = session.graph.blocks["classifier"]
        block.parameters["render_overlay"] = False
        block.name = "SimpleClassify — Rendering disabled control"
        session.save(folder / "no_overlay.json")
    expected, receipt = {}, []
    app = QApplication.instance() or QApplication([])
    for case, count in CASES:
        path = folder / f"{case}.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"]
            window = MainWindow()
            warnings = []
            window.warn = warnings.append
            load_plugin(output / "plugin.py", window.registry)
            window.replace_graph(session.graph, path)
            window.show()
            try:
                block = session.graph.blocks["classifier"]
                for index in range(count):
                    session.run()
                    result = block.last_result
                    assert result is not None and block.state == "pass", block.error
                    before, chart = copy.deepcopy(stable_result(result)), raster_hash(result)
                    expected[f"{case}:{index}"] = before
                    parameters = copy.deepcopy(block.parameters)
                    inspector = window.inspector
                    inspector.select(block)
                    with (
                        patch.object(Classifier, "predict", side_effect=AssertionError("Inferred")),
                        patch.object(
                            dataflow, "load_classifier", side_effect=AssertionError("Loaded")
                        ),
                        patch.object(
                            block, "create_plugin_dialog", side_effect=AssertionError("Trained")
                        ),
                    ):
                        inspector.open_plugin_review_dialog()
                        app.processEvents()
                        popup = inspector._plugin_review_dialog
                        assert popup is not None and popup.snapshot is not None, warnings
                        capture = popup.snapshot
                        assert capture.record.prediction == {name: before[name] for name in PORTS}
                        assert popup.table.rowCount() == len(before["probabilities"])
                        popup.table.selectRow(1)
                        for tab, view in enumerate(popup.views):
                            popup.tabs.setCurrentIndex(tab)
                            view.actual_size()
                            assert view.transform().m11() == 1
                            view.fit_image()
                            assert not view.image_item.pixmap().isNull()
                        exported = None
                        if index in EXPORTS:
                            exported = f"snapshots/{case}_{index}"
                            popup.export_to(output / exported)
                        assert stable_result(result) == before and raster_hash(result) == chart
                        assert block.parameters == parameters
                        receipt.append(
                            dict(
                                case=case,
                                frame=index,
                                source=truth[index]["image"],
                                truth=truth[index]["label"],
                                predicted=before["label"],
                                correct=before["label"] == truth[index]["label"],
                                source_sha256=capture.source_sha256,
                                source_shape=list(capture.source_pixels.shape),
                                source_dtype=capture.source_pixels.dtype.str,
                                executed_assets=capture.record.asset_hashes,
                                runtime_chart_sha256=chart,
                                native_results_action=True,
                                review_did_not_infer_reload_or_train=True,
                                primary_values_and_chart_unchanged=True,
                                parameters_unchanged=True,
                                export=exported,
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
                exports=sum(row["export"] is not None for row in receipt),
                correct={
                    case: sum(r["correct"] for r in receipt if r["case"] == case)
                    for case, _ in CASES
                },
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing public VegQual example folder")
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
