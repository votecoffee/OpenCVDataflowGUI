"""Frozen export integrity, exact numeric/raster replay, native Results and portable Save As."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin

from build_examples import CASES, matcher, raster_hashes, stable_report


def verify_exports(root, originals):
    count = 0
    for row in originals.values():
        for entry in row["exports"]:
            folder = root / entry["path"]
            report = json.loads((folder / "review.json").read_bytes())
            completed = report["completed_execution"]
            source = np.load(folder / "consumed_input.npy", allow_pickle=False)
            assert (
                list(source.shape)
                == row["source_shape"]
                == completed["source_image"]["shape"]
            )
            assert source.dtype.str == row["source_dtype"]
            assert (
                hashlib.sha256(source.tobytes()).hexdigest()
                == row["source_sha256"]
                == completed["source_image"]["sha256"]
            )
            assert (
                hashlib.sha256(
                    (folder / "executed_manifest.json").read_bytes()
                ).hexdigest()
                == row["manifest_sha256"]
            )
            measurement = np.load(folder / "measurement_input.npy", allow_pickle=False)
            assert (
                hashlib.sha256(measurement.tobytes()).hexdigest()
                == report["image_sha256"]
            )
            assert report["station_number"] == entry["station"] + 1
            assert report["group"] == entry["group"]
            assert report["path_passed"] == row["passed"]
            assert (
                not completed["measurement_reexecuted"]
                and not completed["recipe_reloaded"]
            )
            for name in ("overlay", "inspected_mask"):
                image = cv2.imread(str(folder / f"{name}.png"), cv2.IMREAD_UNCHANGED)
                assert image is not None and image.shape[:2] == measurement.shape
            if entry["group"] == "unobserved":
                assert report["trace"]["intensity"] is None
            count += 1
    return count


def verify(root, output, *, core_only=False, native=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    if core_only:
        from opencv_simple_path_width.dataflow import register

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
                    block = matcher(session.graph)
                    assert block.last_result is not None, block.error
                    report = stable_report(block.last_result)
                    assert report == expected[f"{name}:{index}"]
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
                        assert not popup.source_view.image_item.pixmap().isNull()
                        popup.panel._next_issue(1)
                        popup.panel._focus_station()
                        popup.reject()
                        app.processEvents()
                    rows.append(
                        dict(
                            case=name,
                            frame=index,
                            passed=report["passed"],
                            numeric_exact=True,
                            default_rasters_exact=True,
                            native_results_review=window is not None,
                        )
                    )
                if not core_only:
                    target = output / name / "project.json"
                    target.parent.mkdir()
                    session.save(target)
                    relative = matcher(session.graph).parameters["recipe_path"]
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
                    result = matcher(copied.graph).last_result
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
        scope="Exact complete station/scalar/physical reports and default raster bytes; no accuracy claim",
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
