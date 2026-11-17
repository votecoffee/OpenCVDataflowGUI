"""Verify frozen assets/arrays, unchanged numerical/raster replay and native portable projects."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin

from build_examples import CASES, raster_hashes, stable_report


def verify_exports(root, originals):
    for row in originals.values():
        target = root / row["export"]
        report = json.loads((target / "review.json").read_bytes())
        pixels = np.load(target / "consumed_input.npy", allow_pickle=False)
        assert (
            list(pixels.shape) == row["source_shape"] == report["source_image"]["shape"]
        )
        assert pixels.dtype.str == row["source_dtype"]
        assert (
            hashlib.sha256(pixels.tobytes()).hexdigest()
            == row["source_sha256"]
            == report["source_image"]["sha256"]
        )
        for name in ("mask", "inspected"):
            value = np.load(target / f"{name}.npy", allow_pickle=False)
            assert value.shape == pixels.shape[:2]
            assert (
                hashlib.sha256(value.tobytes()).hexdigest()
                == row["runtime_rasters"][name]
            )
        for name, digest in report["exported_assets"].items():
            assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
        assert report["exported_assets"]["executed_model.yml"] == row["model_sha256"]
        if row["region_sha256"] is not None:
            assert (
                report["exported_assets"]["executed_region.png"] == row["region_sha256"]
            )
        if report["scores_available"]:
            votes = np.load(target / "score_map.npy", allow_pickle=False)
            assert votes.shape == pixels.shape[:2]
            assert (
                hashlib.sha256(votes.tobytes()).hexdigest()
                == row["retained_score_sha256"]
            )
            if row["score_source"] == "output":
                assert (
                    hashlib.sha256(votes.tobytes()).hexdigest()
                    == row["runtime_rasters"]["score_map"]
                )
        else:
            assert not (target / "score_map.npy").exists()
        assert report["state"] == row["state"]
        if row["state"] == "no_observed_pixels":
            assert report["source_foreground_fraction"] is None
        assert (
            not report["inference_reexecuted"]
            and not report["recipe_reloaded"]
            and not report["training_changed"]
        )
    return len(originals)


def verify(root, output, *, core_only=False, native=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    if core_only:
        from opencv_defect_segmentation.dataflow import register

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
                load_plugin(root / "plugin.py", window.registry)
                window.replace_graph(session.graph, source)
                window.show()
            try:
                for index in range(frames):
                    session.run()
                    block = session.graph.blocks["segment"]
                    assert block.last_result is not None, block.error
                    assert (
                        stable_report(block.last_result) == expected[f"{name}:{index}"]
                    )
                    assert (
                        raster_hashes(block.last_result)
                        == originals[name, index]["runtime_rasters"]
                    )
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
                        retained = (
                            None
                            if popup.snapshot.scores is None
                            else hashlib.sha256(
                                popup.snapshot.scores.tobytes()
                            ).hexdigest()
                        )
                        assert (
                            retained == originals[name, index]["retained_score_sha256"]
                        )
                        assert (
                            popup.snapshot.report()["state"]
                            == originals[name, index]["state"]
                        )
                        popup.part.setCurrentIndex(1)
                        if popup.table.rowCount():
                            popup.table.selectRow(0)
                        popup.tabs.setCurrentIndex(4)
                        assert not popup.assessment_view.image_item.pixmap().isNull()
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
                    for key in ("model_path", "region_path"):
                        relative = session.graph.blocks["segment"].parameters[key]
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
                    result = copied.graph.blocks["segment"].last_result
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
        scope="Exact completed numerical/mask/score/default raster results; no accuracy qualification",
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
