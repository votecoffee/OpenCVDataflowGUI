"""Actual inspector suggestion/add/Stage/Apply workflows and frozen CAD review clouds."""

import argparse
import copy
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import PointCloud, render_cloud


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def proposal_ready(app, popup):
    deadline = time.monotonic() + 30
    while popup._future is not None and time.monotonic() < deadline:
        app.processEvents()
        popup.poll()
        time.sleep(0.005)
    assert popup._future is None and popup.result is not None, popup.status.text()


def export_proposals(folder, recipe, batch, object_id=None):
    from opencv_simple_bin_picking import proposal_cloud

    write_json(folder / "proposals.json", batch.report())
    selected = [batch.proposals[0]]
    rejected = [p for p in batch.proposals if not p.reviewable]
    if rejected:
        witness = max(rejected, key=lambda p: p.model_collision_count)
        if witness is not selected[0]:
            selected.append(witness)
    if object_id is None:
        graph_folder = folder
    else:
        graph_folder = folder.parent / f"review_object{object_id}"
        (graph_folder / "assets").mkdir(parents=True)
    manifest = []
    with ProjectSession(builtin_registry(), directory=graph_folder) as session:
        for index, item in enumerate(selected):
            graphic = proposal_cloud(recipe, item)
            cloud = PointCloud(
                graphic.points, "m", graphic.frame, colors=graphic.colors, metadata=graphic.metadata
            )
            prefix = "proposal" if index == 0 else "rejected"
            filename = f"{prefix}.ply" if object_id is None else f"assets/{prefix}.ply"
            write_cloud(graph_folder / filename, cloud, binary=True)
            write_json(graph_folder / filename.replace(".ply", ".json"), graphic.metadata)
            assert cv2.imwrite(
                str(graph_folder / f"{prefix}.png"),
                render_cloud(cloud, 1100, 800, yaw=155, pitch=-35, stable_scale=True),
            )
            if object_id is not None:
                block = session.graph.add(
                    session.registry.create(
                        "input.cloud",
                        block_id=prefix,
                        parameters=dict(path=filename, units="m", frame=graphic.frame),
                    )
                )
                block.name = "Model reviewable" if item.reviewable else "Model rejected"
                block.position, block.preview_output = (index * 450, 0), "cloud"
                manifest.append(
                    dict(
                        block=prefix,
                        path=filename,
                        frame=graphic.frame,
                        sha256=hashlib.sha256((graph_folder / filename).read_bytes()).hexdigest(),
                    )
                )
        if object_id is not None:
            session.save(graph_folder / "review.json")
            assert session.inspect()["validation"]["valid"]
            session.run()
            assert all(b.last_result is not None for b in session.graph.blocks.values())
            write_json(graph_folder / "views.json", manifest)


def build(root, output):
    from build_native_training import settle
    from opencv_simple_bin_picking import JawGrasp, JawRecipe, PartMesh, propose_contacts
    from opencv_simple_bin_picking.jaw_persistence import load_jaw_recipe, save_jaw_recipe
    from opencv_simple_bin_picking.jaw_teaching import reference_capture
    from opencv_simple_bin_picking.qt_jaw_proposals import ContactProposalDialog
    from PySide6.QtWidgets import QApplication, QDialog

    from opencv_dataflow.app import MainWindow

    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    app = QApplication.instance() or QApplication([])
    rows = []
    for name, expected in (("clear", 1), ("tless_frame000013", 0)):
        folder = output / name
        folder.mkdir()
        destination = folder / "project.json"
        with ProjectSession.open(
            root / "mesh_training" / name / "project.json", registry
        ) as session:
            session.save(destination)
            session.run()
            block = session.graph.blocks["jaw"]
            original = copy.deepcopy(block.parameters)
            before = load_jaw_recipe(folder / block.parameters["recipe_path"])
            window, children, batches = MainWindow(), [], []

            def cancel_review(popup):
                children.append(popup)
                proposal_ready(app, popup)
                popup.reject()
                return QDialog.DialogCode.Rejected

            def add_one(popup):
                children.append(popup)
                proposal_ready(app, popup)
                batches.append(popup.result)
                index = next(i for i, p in enumerate(popup.result.proposals) if p.reviewable)
                popup.table.selectRow(index)
                assert popup.viewer.cloud.metadata["acceptance_applied"] is False
                popup.viewer.flyaround.setChecked(True)
                yaw = popup.viewer.yaw.value()
                popup.viewer.advance_orbit()
                assert popup.viewer.yaw.value() != yaw
                popup.choose()
                assert len(popup.selected) == 1
                return QDialog.DialogCode.Accepted

            try:
                load_plugin(root / "plugin.py", window.registry)
                window.replace_graph(session.graph, destination)
                window.scene.nodes["jaw"].setSelected(True)
                app.processEvents()
                inspector = window.inspector
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                existing = dialog.session.recipe
                with patch.object(ContactProposalDialog, "exec", cancel_review):
                    dialog.suggest_contacts()
                assert dialog.session.recipe is existing
                dialog.reject()
                dialog.pool.shutdown(wait=True)
                assert block.parameters == original and inspector.read_parameters() == original
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                with patch.object(ContactProposalDialog, "exec", add_one):
                    dialog.suggest_contacts()
                assert len(dialog.session.recipe.grasps) == len(before.grasps) + 1
                assert not dialog.session.ready and not dialog.buttons["Stage recipe"].isEnabled()
                dialog.preview_reference()
                settle(app, dialog)
                assert dialog.session.ready and len(dialog.session.result.picks) == expected
                dialog.stage()
                dialog.pool.shutdown(wait=True)
                assert block.parameters == original
                assert inspector.apply()
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                after = dialog.session.recipe
                np.testing.assert_array_equal(after.part.points, before.part.points)
                np.testing.assert_array_equal(after.mesh.triangles, before.mesh.triangles)
                assert reference_capture(after).capture_id == reference_capture(before).capture_id
                assert len(after.grasps) == len(before.grasps) + 1
                assert len(after.notes["jaw_contact_suggestions_v1"]) == 1
                dialog.preview_reference()
                settle(app, dialog)
                assert len(dialog.session.result.picks) == expected
                dialog.reject()
                dialog.pool.shutdown(wait=True)
                session.save(destination)
            finally:
                window.dirty = False
                window.close()
                for child in children:
                    child.pool.shutdown(wait=True)
                app.processEvents()
        export_proposals(folder, before, batches[0], 6 if name != "clear" else None)
        with ProjectSession.open(destination, registry) as reopened:
            reopened.run()
            block = reopened.graph.blocks["jaw"]
            assert block.last_result is not None, block.error
            result = block.last_result.outputs
            assert result["count"] == expected
            write_json(folder / "expected.json", result["report"])
            for key in ("graphics", "mesh_fit"):
                cloud = result[key]
                write_cloud(folder / f"{key}.ply", cloud, binary=True)
                write_json(folder / f"{key}.json", cloud.metadata)
                assert cv2.imwrite(
                    str(folder / f"{key}.png"),
                    render_cloud(
                        cloud,
                        1100,
                        800,
                        yaw=155,
                        pitch=-35,
                        stable_scale=True,
                    ),
                )
            shutil.copy2(folder / "graphics.png", folder / "result.png")
        rows.append(
            dict(
                case=name,
                eligible=expected,
                model_reviewable=sum(p.reviewable for p in batches[0].proposals),
                actual_inspector_cancel_add_stage_apply_reopen=True,
                retained_reference_and_geometry_exact=True,
                source_assistance="Authored pose" if name == "clear" else "T-LESS supplied pose",
                robot_execution=False,
            )
        )
    # A second real CAD shape demonstrates rejected suggestions with exactly the
    # same gripper settings. Its reference is explicitly empty: CAD-only review.
    mesh = PartMesh.read(
        root / "pose_agreement/source/models_cad/obj_000008.ply", scale=0.001, frame="tless8"
    )
    part = mesh.sample()
    low, high = part.points.min(0), part.points.max(0)
    first, second = (low + high) / 2, (low + high) / 2
    first[0], second[0] = low[0], high[0]
    seed = JawGrasp.from_contacts("envelope_seed", first, second, [0, 0, 1])
    recipe = JawRecipe(
        replace(
            before.bin,
            reference_depth=np.zeros_like(before.bin.reference_depth),
            reference_instances=None,
            reference_rgb=None,
        ),
        part,
        (seed,),
        before.settings,
        mesh=mesh,
    )
    batch = propose_contacts(recipe)
    assert batch.proposals and not any(p.reviewable for p in batch.proposals)
    folder = output / "object8_rejected"
    folder.mkdir()
    save_jaw_recipe(folder / "cad_only.sbpj", recipe)
    export_proposals(folder, recipe, batch, 8)
    rows.append(
        dict(
            case="object8_rejected",
            model_reviewable=0,
            cad_only=True,
            no_reference_pose_or_measurements=True,
            added_to_training=False,
            runtime_acceptance_asserted=False,
        )
    )
    write_json(output / "checks.json", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    registry = builtin_registry()
    load_plugin(args.root.resolve() / "plugin.py", registry)
    build(args.root.resolve(), args.output.resolve())
