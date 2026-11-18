"""Replay numerical results/charts, frozen exports, native review and portable Save As."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from build_examples import CASES, raster_hash, stable_result
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin


def verify(root, output, *, core_only=False, native=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    if core_only:
        from opencv_simple_classifier.dataflow import ClassifyImage, register

        registry.types.pop(ClassifyImage.type_id, None)
        register(registry)
    else:
        load_plugin(root / "plugin.py", registry)
    app = None
    if native:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    expected = json.loads((root / "expected.json").read_bytes())
    originals = {
        (r["case"], r["frame"]): r
        for r in json.loads((root / "native-verification.json").read_bytes())
    }
    exported = 0
    for row in originals.values():
        if row["export"] is None:
            continue
        target = root / row["export"]
        report = json.loads((target / "review.json").read_bytes())
        image = np.load(target / "consumed_input.npy", allow_pickle=False)
        assert hashlib.sha256(image.tobytes()).hexdigest() == row["source_sha256"]
        assert list(image.shape) == row["source_shape"] and image.dtype.str == row["source_dtype"]
        for name, digest in row["executed_assets"].items():
            assert (
                hashlib.sha256((target / "executed_recipe" / name).read_bytes()).hexdigest()
                == digest
            )
        assert report["prediction"] == {
            k: v for k, v in expected[f"{row['case']}:{row['frame']}"].items() if k != "passed"
        }
        assert not report["inference_reexecuted"] and not report["training_changed"]
        exported += 1
    rows = []
    for case, count in CASES:
        path = root / "vegqual" / f"{case}.json"
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"]
            block = session.graph.blocks["classifier"]
            assets = (
                None
                if core_only
                else {name: (path.parent / name).read_bytes() for name in block.project_assets()}
            )
            window = None
            if app is not None:
                from opencv_dataflow.app import MainWindow

                window = MainWindow()
                load_plugin(root / "plugin.py", window.registry)
                window.replace_graph(session.graph, path)
                window.show()
            try:
                for index in range(count):
                    session.run()
                    assert block.last_result is not None, block.error
                    assert stable_result(block.last_result) == expected[f"{case}:{index}"]
                    assert (
                        raster_hash(block.last_result)
                        == originals[case, index]["runtime_chart_sha256"]
                    )
                    if window is not None:
                        window.inspector.select(block)
                        window.inspector.open_plugin_review_dialog()
                        app.processEvents()
                        popup = window.inspector._plugin_review_dialog
                        assert popup is not None and popup.snapshot is not None
                        assert (
                            popup.snapshot.source_sha256 == originals[case, index]["source_sha256"]
                        )
                        assert (
                            popup.snapshot.record.asset_hashes
                            == originals[case, index]["executed_assets"]
                        )
                        popup.table.selectRow(1)
                        popup.tabs.setCurrentIndex(1)
                        assert not popup.assessment_view.image_item.pixmap().isNull()
                        popup.reject()
                        app.processEvents()
                    rows.append(
                        dict(
                            case=case,
                            frame=index,
                            numeric_exact=True,
                            default_chart_exact=True,
                            native_results_review=window is not None,
                        )
                    )
                if not core_only:
                    target = output / case / "project.json"
                    target.parent.mkdir()
                    session.save(target)
                    for name, content in assets.items():
                        assert (target.parent / name).read_bytes() == content
            finally:
                if window is not None:
                    window.dirty = False
                    window.close()
                    app.processEvents()
        if not core_only:
            with ProjectSession.open(target, registry) as copied:
                assert copied.inspect()["validation"]["valid"]
                for index in range(count):
                    copied.run()
                    result = copied.graph.blocks["classifier"].last_result
                    assert (
                        result is not None and stable_result(result) == expected[f"{case}:{index}"]
                    )
                    assert raster_hash(result) == originals[case, index]["runtime_chart_sha256"]
            for row in rows[-count:]:
                row["portable_save_as"] = True
    receipt = dict(
        core_only=core_only,
        native=native,
        exported_reviews_verified=exported,
        cases=rows,
        scope=(
            "Exact completed scores and default chart bytes; no independent accuracy qualification"
        ),
    )
    (output / "checks.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                frames=len(rows),
                native=native,
                exports=exported,
                numerical_results_exact=True,
                default_charts_exact=True,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    verify(args.root.resolve(), args.output.resolve(), core_only=args.core_only, native=args.native)
