"""Verify exact native arrays/model exports, runtime replay and portable Save As."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin

from build_examples import CASES, MAPS, load_companion, raster_hashes, stable_report


def verify_exports(root, originals):
    for row in originals.values():
        folder = root / row["export"]
        report = json.loads((folder / "review.json").read_bytes())
        completed = report["completed_execution"]
        source = np.load(folder / "consumed_input.npy", allow_pickle=False)
        assert (
            list(source.shape)
            == row["source_shape"]
            == completed["source_image"]["shape"]
        )
        assert source.dtype.str == row["source_dtype"]
        assert hashlib.sha256(source.tobytes()).hexdigest() == row["source_sha256"]
        assert completed["original_recipe_sha256"] == row["original_recipe_sha256"]
        assert (
            hashlib.sha256((folder / "executed_model.sri").read_bytes()).hexdigest()
            == completed["exported_model"]["sha256"]
        )
        for name in MAPS:
            reference = np.load(folder / f"{name}.npy", allow_pickle=False)
            output = np.load(folder / f"source_{name}.npy", allow_pickle=False)
            assert list(reference.shape[::-1]) == row["reference_size"]
            assert output.shape == source.shape[:2]
            assert (
                hashlib.sha256(reference.tobytes()).hexdigest()
                == report["arrays"][name]["sha256"]
            )
            assert (
                hashlib.sha256(output.tobytes()).hexdigest()
                == row["runtime_rasters"][name]
            )
        assert report["state"] == row["state"]
        truth = row.get("ground_truth")
        if truth is not None and "comparison" in truth:
            assert (folder / truth["comparison"]).is_file()
        if row["state"] == "not_evaluated":
            assert (
                report["metrics"][3]["value"] is None
                and report["inspection"]["evaluated"] is False
            )
        assert (
            not completed["inspection_reexecuted"]
            and not completed["exposure_refitted"]
        )
        assert not completed["recipe_reloaded"] and not completed["training_changed"]
    return len(originals)


def verify(root, output, *, core_only=False, native=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    load_companion(root, registry)
    if core_only:
        from opencv_simple_reference_inspect.dataflow import register

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
    exported_reviews = verify_exports(root, originals)
    rows = []
    for name, _, frames in CASES:
        source = root / name / "project.json"
        with ProjectSession.open(source, registry) as session:
            assert session.inspect()["validation"]["valid"]
            window = None
            if app is not None:
                from opencv_dataflow.app import MainWindow

                window = MainWindow()
                load_companion(root, window.registry)
                load_plugin(root / "plugin.py", window.registry)
                window.replace_graph(session.graph, source)
                window.show()
            try:
                for index in range(frames):
                    session.run()
                    block = session.graph.blocks["inspection"]
                    assert block.last_result is not None, block.error
                    report = stable_report(block.last_result)
                    assert report == expected[f"{name}:{index}"], (
                        name,
                        index,
                        "numerical report differs",
                    )
                    assert (
                        raster_hashes(block.last_result)
                        == originals[name, index]["runtime_rasters"]
                    ), (name, index, "raster differs")
                    if window is not None:
                        window.inspector.select(block)
                        window.inspector.open_plugin_review_dialog()
                        app.processEvents()
                        popup = window.inspector._plugin_review_dialog
                        assert popup is not None and popup.snapshot is not None
                        assert (
                            popup.snapshot.source_sha256
                            == originals[name, index]["source_sha256"]
                        )
                        assert (
                            popup.panel.review.summary()["state"]
                            == originals[name, index]["state"]
                        )
                        popup.panel.mode.setCurrentText("Inspected coverage")
                        popup.panel.tabs.setCurrentIndex(2)
                        assert not popup.panel.sheet.image_item.pixmap().isNull()
                        popup.reject()
                        app.processEvents()
                    rows.append(
                        dict(
                            case=name,
                            frame=index,
                            state=originals[name, index]["state"],
                            numeric_exact=True,
                            default_rasters_exact=True,
                            native_results_review=window is not None,
                        )
                    )
                if not core_only:
                    target = output / name / "project.json"
                    target.parent.mkdir()
                    session.save(target)
                    for block in session.graph.blocks.values():
                        relative = block.parameters.get("recipe_path")
                        if relative:
                            assert (target.parent / relative).read_bytes() == (
                                source.parent / relative
                            ).read_bytes()
            finally:
                if window is not None:
                    window.dirty = False
                    window.close()
                    app.processEvents()
        if not core_only:
            with ProjectSession.open(target, registry) as copied:
                assert copied.inspect()["validation"]["valid"]
                for index in range(frames):
                    copied.run()
                    result = copied.graph.blocks["inspection"].last_result
                    assert (
                        result is not None
                        and stable_report(result) == expected[f"{name}:{index}"]
                    )
                    assert (
                        raster_hashes(result)
                        == originals[name, index]["runtime_rasters"]
                    )
            for row in rows[-frames:]:
                row["portable_save_as"] = True
    receipt = dict(
        core_only=core_only,
        native=native,
        exported_reviews_verified=exported_reviews,
        cases=rows,
        scope="Complete reports and reference/source/default raster bytes; real-data errors remain, no accuracy claim",
    )
    (output / "checks.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                frames=len(rows),
                native=native,
                exported_reviews=exported_reviews,
                numerical_reports_exact=True,
                default_rasters_exact=True,
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
    verify(
        args.root.resolve(),
        args.output.resolve(),
        core_only=args.core_only,
        native=args.native,
    )
