"""Actual pinned plugin/worker/native review; no desktop input or runtime-profile writes."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.assistant_api import make_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugin_dialog import PluginDialogContext
from PySide6.QtWidgets import QApplication


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def ply_arrays(path):
    header, data = path.read_bytes().split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    values = np.frombuffer(data, dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    return values["xyz"], values["rgb"]


def run(
    plugin, worker, models, output, cases, *, global_search=False, summary_only=False, baseline=None
):
    os.environ["SIMPLE_FOUNDATION_POSE_PYTHON"] = str(worker.resolve())
    os.environ["SIMPLE_FOUNDATION_POSE_MODELS"] = str(models.resolve())
    app = QApplication.instance() or QApplication([])
    registry = make_registry(plugins=[str(plugin.resolve())])
    from opencv_simple_foundation_pose.qt_completed_review import CompletedReviewDialog

    output.mkdir(parents=True, exist_ok=False)
    original = plugin.parent / "worker_execution"
    portable = not original.is_dir()
    receipts = []
    for case in cases:
        folder = output / case
        folder.mkdir()
        for source, target in (
            ("trained.sfp", "trained.sfp"),
            (case + ".npz", "capture.npz"),
            (case + "_prior_proposals.json", "coarse.json"),
        ):
            location = plugin.parent / case / target if portable else original / "assets" / source
            shutil.copyfile(location, folder / target)
        project_path = (
            plugin.parent / case / "run_project.json"
            if portable
            else original / (case + "_assisted_replay.json")
        )
        project = json.loads(project_path.read_bytes())
        source = next(b for b in project["blocks"] if b["id"] == "capture")
        source["parameters"].update(
            capture_path="capture.npz", coarse_path="" if global_search else "coarse.json"
        )
        tool = next(b for b in project["blocks"] if b["id"] == "pose")
        tool["parameters"]["recipe_path"] = "trained.sfp"
        if global_search:
            project["connections"] = [
                c for c in project["connections"] if c["input"] != "coarse_poses"
            ]
            tool["description"] = (
                "Global rotation grid on supplied instance mask; no coarse or pose-label input."
            )
        execution = Graph.from_dict(project, registry, project_dir=folder)
        display = context = dialog = None
        try:
            start = time.perf_counter()
            execution.run()
            block = execution.blocks["pose"]
            assert block.last_result is not None, block.error
            report = block.last_result.outputs["report"]
            assert report["execution"]["score_calls"] == 1
            expected_count = 252 if global_search else 2
            assert len(report["candidates"]) == expected_count
            client, process = block._client, block._client.process
            display = Graph.from_dict(execution.to_dict(), registry, project_dir=folder)
            display.copy_runtime_from(execution)
            for key, b in execution.blocks.items():
                display.blocks[key].copy_results_from(b)
            execution.close()
            assert process.poll() is None
            shown = display.blocks["pose"]
            context = PluginDialogContext(display, shown)
            dialog = CompletedReviewDialog(
                None, context, lambda: shown.prepare_completed_review(context)
            )
            dialog.capture_latest()
            deadline = time.monotonic() + 120
            while dialog.pending is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            assert dialog.frozen is not None, dialog.status.text()
            frozen = dialog.frozen
            assert report["candidates"] == frozen.report["candidates"]
            assert dialog.table.rowCount() == expected_count
            for index in sorted({0, 1, expected_count - 1}):
                dialog.table.selectRow(index)
                assert dialog.cloud.flyaround.isEnabled()
                dialog.cloud.advance_orbit()
                if dialog.sample.isEnabled():
                    dialog.sample.setValue(min(15, dialog.sample.maximum()))
            if global_search:
                assert baseline is not None
                baseline_case = baseline / case if (baseline / case).is_dir() else baseline
                expected = json.loads((baseline_case / "report.json").read_bytes())
                assert report["candidates"] == expected["candidates"], (
                    "Historical global cohort differs"
                )
                np.testing.assert_array_equal(
                    frozen.graphics["overlay_rgb"],
                    cv2.imread(str(baseline_case / "overlay.png"))[..., ::-1],
                )
                xyz, rgb = ply_arrays(baseline_case / "fit.ply")
                np.testing.assert_array_equal(frozen.graphics["review_xyz"], xyz)
                np.testing.assert_array_equal(frozen.graphics["review_rgb"], rgb)
            else:
                # Same exact input/settings/cohort, with diagnostic retention disabled.
                arrays = {k: frozen.arrays[k] for k in ("rgb", "depth", "mask", "camera")}
                arrays["initial"] = frozen.arrays["supplied_initial"]
                result, graphics = client.call(
                    "register",
                    dict(
                        recipe_path=str((folder / "trained.sfp").resolve()),
                        recipe_sha256=shown.parameters["sha256"],
                        weights_dir=str(models.resolve()),
                        capture_id=report["capture_id"],
                        frame=report["target_frame"],
                        graphics=True,
                    ),
                    arrays,
                )
                assert result["report"]["candidates"] == report["candidates"]
                for key in frozen.graphics:
                    np.testing.assert_array_equal(graphics[key], frozen.graphics[key])
                # A stale read must not retire the newer cached worker.
                try:
                    client.call("capture_completed", dict(identity=frozen.record.identity))
                except ValueError as error:
                    assert "expired" in str(error)
                else:
                    raise AssertionError("Expected expired diagnostic token")
                assert client.process is process and process.poll() is None
            display.close()
            assert process.poll() is not None
            dialog.table.selectRow(0)
            dialog.show_candidate()
            if not summary_only:
                frozen.export(folder / "frozen")
            else:
                write(folder / "report.json", frozen.report)
                for index in (0, 1, expected_count - 1):
                    cv2.imwrite(
                        str(folder / f"rank_{index + 1:03d}.png"), frozen.image(index)[..., ::-1]
                    )
            write(folder / "run_project.json", project)
            receipt = dict(
                case=case,
                initialization=report["execution"]["initialization"],
                candidates=expected_count,
                accepted=report["accepted_candidates"],
                complete_cohort_score_calls=1,
                original_candidates_and_graphics_exact=True,
                native_review_after_execution_graph_closed=True,
                review_and_export_after_worker_exit=True,
                expired_read_preserves_worker=None if global_search else True,
                full_frozen_export=not summary_only,
                captured_sample_counts=[len(e.get("points", ())) for e in frozen.candidates],
                transferred_fit_bytes=sum(v.nbytes for e in frozen.candidates for v in e.values()),
                elapsed_seconds=time.perf_counter() - start,
                scope="Visible mask; two prior network seeds when assisted; no pose truth input.",
                recipe_sha256=hashlib.sha256(frozen.recipe_bytes).hexdigest(),
                source="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
            )
            write(folder / "validation.json", receipt)
            receipts.append(receipt)
            print(
                f"{case}: {expected_count} candidates / {receipt['accepted']} accepted; "
                "original cohort, fit and graphics exact",
                flush=True,
            )
        finally:
            if dialog is not None:
                dialog.reject()
            if context is not None:
                context._close()
            if display is not None:
                display.close()
            execution.close()
    write(output / "validation.json", dict(cases=receipts, weights_bundled=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plugin", "worker", "models", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--case", action="append", choices=("held_out_frame13", "different_object_negative")
    )
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    run(
        args.plugin,
        args.worker,
        args.models,
        args.output,
        args.case or ("held_out_frame13", "different_object_negative"),
        global_search=args.global_search,
        summary_only=args.summary_only,
        baseline=args.baseline,
    )
