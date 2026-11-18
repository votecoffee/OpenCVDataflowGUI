"""Actual-worker/native review with graph handoff, numerical parity and frozen export.

Use the host's Python, with GDRNPP/src on PYTHONPATH for the source adapter, or
--host-adapter for the published adapter. No desktop input or global runtime edits.
"""

import argparse
import copy
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from opencv_dataflow.assistant_api import make_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugin_dialog import PluginDialogContext


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def stable(report):
    return {
        k: copy.deepcopy(report[k])
        for k in ("best_pose", "accepted_candidates", "candidates", "fit_settings", "setup")
    }


def run(recipes, worker_python, models, output, *, host_adapter=False, connected=False):
    package = "opencv_dataflow.simple_gdrnpp" if host_adapter else "opencv_simple_gdrnpp"
    adapter = importlib.import_module(package + ".dataflow")
    bridge = importlib.import_module(package + ".bridge")
    qt = importlib.import_module(package + ".qt_completed_review")
    output.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    registry = make_registry()
    adapter.register(registry)
    receipts = []
    for number, recipe in enumerate(recipes, 1):
        folder = output / f"case_{number:02d}"
        folder.mkdir()
        raw = recipe.read_bytes()
        (folder / "input.gdrnpp").write_bytes(raw)
        params = dict(
            recipe_path="input.gdrnpp",
            sha256=hashlib.sha256(raw).hexdigest(),
            models_dir=str(models.resolve()),
            mode="Connected RGB-D" if connected else "Retained reference",
            render_overlay=True,
            render_cloud=True,
            render_gallery=False,
        )
        execution = Graph()
        execution.bind_project_dir(folder)
        block = execution.add(adapter.SimpleGDRNPP(block_id="locate", parameters=params))
        client = block._client = bridge.WorkerClient(python=str(worker_python.resolve()))
        display = context = dialog = None
        try:
            inputs, baseline_current, baseline_arrays = {}, {}, {}
            if connected:
                from opencv_dataflow.images import DepthArray, ImageArray

                summary, reference = client.call(
                    "inspect",
                    dict(
                        base=dict(
                            kind="recipe",
                            path=str((folder / "input.gdrnpp").resolve()),
                            sha256=params["sha256"],
                        )
                    ),
                )
                setup = summary["setup"]
                inputs = dict(
                    image=ImageArray(reference["rgb"], "rgb"),
                    depth=DepthArray(reference["depth"].astype(np.float64) * 1000, "mm"),
                    camera=dict(
                        matrix=reference["camera"].tolist(),
                        image_size=setup["image_size"],
                        frame=setup["camera_frame"],
                    ),
                    capture_id=setup["reference_capture_id"] + "_connected",
                    detections=[
                        {**d, "box": [int(v) for v in d["box"]]}
                        for d in setup["reference_detections"]
                    ],
                )
                current, baseline_arrays = adapter.connected_capture(inputs, {})
                baseline_current = dict(current=current)
            started = time.perf_counter()
            assert block.execute(inputs) is not None, block.error
            seconds = time.perf_counter() - started
            process = client.process
            display = Graph.from_dict(execution.to_dict(), registry, project_dir=folder)
            display.copy_runtime_from(execution)
            shown = display.blocks["locate"]
            shown.copy_results_from(block)
            execution.close()
            assert client.process is process and process.poll() is None
            context = PluginDialogContext(display, shown)
            dialog = qt.CompletedReviewDialog(
                None, context, lambda: shown.prepare_completed_review(context)
            )
            dialog.capture_latest()
            deadline = time.monotonic() + 60
            while dialog.pending is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            assert dialog.frozen is not None, dialog.status.text()
            frozen, report = dialog.frozen, dialog.frozen.report
            assert report["worker"]["inference_request_count"] == 1
            assert dialog.cloud.cloud is not None and dialog.cloud.flyaround.isEnabled()
            for i in range(len(report["candidates"])):
                dialog.table.selectRow(i)
                assert dialog.index() == i and dialog.cloud.cloud is not None
                if dialog.sample.isEnabled():
                    dialog.sample.setValue(min(10, dialog.sample.maximum()))
            baseline, arrays = client.call(
                "preview",
                dict(
                    base=dict(
                        kind="recipe",
                        path=str((folder / "input.gdrnpp").resolve()),
                        sha256=params["sha256"],
                    ),
                    models_dir=str(models.resolve()),
                    graphics=True,
                    want_image=True,
                    want_cloud=True,
                    retain_completed=False,
                    **baseline_current,
                ),
                baseline_arrays,
            )
            assert stable(baseline["report"]) == stable(report)
            for key in frozen.graphics:
                np.testing.assert_array_equal(arrays[key], frozen.graphics[key])
            assert process.poll() is None
            display.close()
            assert process.poll() is not None and client.process is None
            dialog.table.selectRow(0)
            dialog.show_candidate()
            frozen.export(folder / "frozen")
            assert (folder / "frozen/executed.gdrnpp").read_bytes() == raw
            project = display.to_dict()
            project["blocks"][0]["parameters"]["models_dir"] = ""
            write(folder / "run_project.json", project)
            write(
                folder / "review_project.json",
                dict(
                    schema_version=1,
                    blocks=[
                        dict(
                            id=f"stored_{i}",
                            type="input.cloud",
                            name=f"Recorded proposal {i + 1}: {c['fit']['status']}",
                            parameters=dict(
                                path=f"frozen/candidate_{i + 1:02d}.ply",
                                units="m",
                                frame=report["target_frame"],
                                max_points=1000000,
                            ),
                            position=[i * 420, 0],
                            thumbnail=True,
                            preview_output="cloud",
                            exposed_inputs=["ticket"],
                            exposed_outputs=["cloud", "count", "ticket"],
                        )
                        for i, c in enumerate(report["candidates"])
                    ],
                    connections=[],
                ),
            )
            receipt = dict(
                case=folder.name,
                capture_id=report["capture_id"],
                proposals=len(report["candidates"]),
                accepted=report["accepted_candidates"],
                selected_proposal=frozen.bundle.outputs["review"].value["candidate_number"],
                archive_sha256=params["sha256"],
                inference_requests_through_review=1,
                baseline_same_settings_numeric_and_graphics_identical=True,
                runner_graph_handoff_preserved_worker=True,
                retired_worker_pid=process.pid,
                retired_before_browsing_and_export=True,
                verification_samples=[len(e.get("points", ())) for e in frozen.candidates],
                transferred_fit_array_bytes=sum(
                    v.nbytes for e in frozen.candidates for v in e.values()
                ),
                cold_execution_seconds=seconds,
                source="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
                mode=params["mode"],
                scope=(
                    "Retained-reference demonstration, supplied boxes; "
                    "no pose labels used in inference"
                ),
            )
            receipts.append(receipt)
            write(folder / "validation.json", receipt)
            print(
                f"{folder.name}: {receipt['accepted']}/{receipt['proposals']} accepted; "
                "native review, exact parity, retired worker",
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
            client.close()
    write(
        output / "validation.json", dict(schema="simple_gdrnpp_completed_checks/v1", cases=receipts)
    )
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, nargs="+", required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-adapter", action="store_true")
    parser.add_argument("--connected", action="store_true")
    args = parser.parse_args()
    run(
        args.recipes,
        args.worker_python,
        args.models,
        args.output,
        host_adapter=args.host_adapter,
        connected=args.connected,
    )
