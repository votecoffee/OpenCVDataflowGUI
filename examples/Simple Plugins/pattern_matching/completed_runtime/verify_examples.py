"""Numerical replay, native read-only review and portable Save As; optional old-core comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin

from build_examples import CASES, stable_report


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        return max(
            (compare(actual[key], value) for key, value in expected.items()),
            default=0.0,
        )
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        return max((compare(a, b) for a, b in zip(actual, expected)), default=0.0)
    if type(expected) is float:
        assert abs(actual - expected) <= 1e-5, (actual, expected)
        return abs(actual - expected)
    assert actual == expected, (actual, expected)
    return 0.0


def verify_exports(root, originals):
    count = 0
    for original in originals.values():
        for relative in original["exports"]:
            folder = root / relative
            report = json.loads((folder / "review.json").read_bytes())
            completed = report["completed_execution"]
            pixels = np.load(folder / "consumed_input.npy", allow_pickle=False)
            source = completed["source_image"]
            assert list(pixels.shape) == source["shape"]
            assert pixels.dtype.str == source["dtype"]
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == source["sha256"]
            assert source["sha256"] == original["source_sha256"]
            assert (
                completed["executed_model"]["recipe_sha256"] == original["model_sha256"]
            )
            assert (
                hashlib.sha256(
                    (folder / "executed_recipe.npz").read_bytes()
                ).hexdigest()
                == (completed["exported_recipe"]["sha256"])
            )
            for image_name in ("fit", "working", "inspected", "inspected_mask"):
                image = cv2.imread(
                    str(folder / f"{image_name}.png"), cv2.IMREAD_UNCHANGED
                )
                assert image is not None
                assert list(image.shape[1::-1]) == completed["working_size"]
            assert completed["accepted"] == original["accepted"]
            if original["accepted"] == 0:
                assert report["selected"] is None
            assert not completed["search_reexecuted"]
            assert not completed["recipe_reloaded"]
            assert not completed["training_changed"]
            count += 1
    return count


def verify(root, output, *, core_only=False, native=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    if core_only:
        from opencv_geometric_pattern_match.dataflow import register

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
    for name, _, _, frames in CASES:
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
                    block = session.graph.blocks["matcher"]
                    assert block.last_result is not None, block.error
                    report = stable_report(block.last_result)
                    error = compare(report, expected[f"{name}:{index}"])
                    native_passed = False
                    if window is not None:
                        inspector = window.inspector
                        inspector.select(block)
                        inspector.open_plugin_review_dialog()
                        app.processEvents()
                        popup = inspector._plugin_review_dialog
                        assert popup is not None and popup.snapshot is not None
                        assert (
                            popup.snapshot.source_sha256
                            == originals[name, index]["source_sha256"]
                        )
                        assert len(popup.snapshot.record.matches) == report["count"]
                        assert (
                            not popup.fit_view.image.isNull()
                            and not popup.source_view.image.isNull()
                        )
                        popup.overview()
                        popup.reject()
                        app.processEvents()
                        native_passed = True
                    overlay_hash = hashlib.sha256(
                        np.asarray(block.last_result.outputs["overlay"]).tobytes()
                    ).hexdigest()
                    rows.append(
                        dict(
                            case=name,
                            frame=index,
                            count=report["count"],
                            maximum_numeric_difference=error,
                            default_overlay_exact=overlay_hash
                            == originals[name, index]["runtime_overlay_sha256"],
                            native_completed_review=native_passed,
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
                    result = copied.graph.blocks["matcher"].last_result
                    assert result is not None
                    compare(stable_report(result), expected[f"{name}:{index}"])
            for row in rows[-frames:]:
                row["portable_save_as"] = True
    receipt = dict(
        core_only=core_only,
        native=native,
        cases=rows,
        numerical_tolerance=1e-5,
        exported_review_integrity_checks=exported_reviews,
        scope="Repeated solver comparisons; frozen review itself changes no numerical values or overlay bytes",
    )
    (output / "checks.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                frames=len(rows),
                native=native,
                maximum_numeric_difference=max(
                    row["maximum_numeric_difference"] for row in rows
                ),
                default_overlays_exact=sum(
                    row["default_overlay_exact"] for row in rows
                ),
                exported_reviews=exported_reviews,
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
