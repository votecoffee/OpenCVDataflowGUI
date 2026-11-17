"""Train two retained triangle-CAD examples through actual inspector Stage/Apply."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import render_cloud


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def box_stl(path):
    faces = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            points = np.zeros((4, 3))
            points[:, axis] = sign * 20
            points[:, others] = [[-20, -20], [-20, 20], [20, 20], [20, -20]]
            for ids in ([0, 1, 2], [0, 2, 3]):
                face = points[ids]
                if np.cross(face[1] - face[0], face[2] - face[0])[axis] * sign < 0:
                    face = face[[0, 2, 1]]
                faces.append(face)
    text = ["solid authored_box"]
    for face in faces:
        text.extend(["facet normal 0 0 0", "outer loop"])
        text.extend("vertex " + " ".join(map(str, row)) for row in face)
        text.extend(["endloop", "endfacet"])
    path.write_text("\n".join(text + ["endsolid authored_box"]) + "\n", encoding="ascii")


def build(root, output):
    from build_native_training import settle
    from opencv_simple_bin_picking import PartMesh
    from opencv_simple_bin_picking.jaw_persistence import load_jaw_recipe
    from opencv_simple_bin_picking.jaw_teaching import reference_capture
    from opencv_simple_bin_picking.persistence import save_recipe
    from PySide6.QtWidgets import QApplication, QInputDialog

    from opencv_dataflow.app import MainWindow

    output.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    rows = []
    for name, source, expected in (
        ("clear", "clear", 1),
        ("tless_frame000013", "tless/frame000013", 0),
    ):
        folder = output / name
        (folder / "input").mkdir(parents=True)
        destination = folder / "project.json"
        with ProjectSession.open(root / source / "project.json", registry) as session:
            session.save(destination)
            session.run()
            block = session.graph.blocks["jaw"]
            old = load_jaw_recipe(folder / block.parameters["recipe_path"])
            save_recipe(folder / "input/bin.sbp", old.bin)
            cad = folder / ("input/box.stl" if name == "clear" else "input/object6.ply")
            if name == "clear":
                box_stl(cad)
            else:
                cad.write_bytes((root / "tless/source/models_cad/obj_000006.ply").read_bytes())
            original = copy.deepcopy(block.parameters)
            window = MainWindow()
            try:
                load_plugin(root / "plugin.py", window.registry)
                window.replace_graph(session.graph, destination)
                window.scene.nodes["jaw"].setSelected(True)
                app.processEvents()
                inspector = window.inspector
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                with (
                    patch.object(dialog, "choose", side_effect=[folder / "input/bin.sbp", cad]),
                    patch.object(
                        QInputDialog, "getItem", side_effect=[("mm", True), ("File winding", True)]
                    ),
                    patch.object(QInputDialog, "getText", return_value=(old.part.frame, True)),
                ):
                    dialog.choose_mesh_setup()
                assert dialog.session.recipe.mesh is not None
                np.testing.assert_array_equal(
                    dialog.session.recipe.mesh.triangles,
                    PartMesh.read(cad, scale=0.001, frame=old.part.frame).triangles,
                )
                dialog.reject()
                dialog.pool.shutdown(wait=True)
                assert block.parameters == original and inspector.read_parameters() == original
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                with (
                    patch.object(dialog, "choose", side_effect=[folder / "input/bin.sbp", cad]),
                    patch.object(
                        QInputDialog, "getItem", side_effect=[("mm", True), ("File winding", True)]
                    ),
                    patch.object(QInputDialog, "getText", return_value=(old.part.frame, True)),
                ):
                    dialog.choose_mesh_setup()
                notes = dict(dialog.session.recipe.notes)
                notes.update(old.notes)
                dialog.session.change(notes=notes)
                triangles = dialog.session.recipe.mesh.triangles.copy()
                points = dialog.session.recipe.part.points.copy()
                dialog.capture_reference()
                capture = reference_capture(dialog.session.recipe)
                assert capture is not None and capture.identity["snapshot_id"]
                dialog.preview_reference()
                settle(app, dialog)
                assert dialog.session.ready, dialog.status.text()
                assert len(dialog.session.result.picks) == expected
                assert dialog.session.result.mesh_fits[0].summary["acceptance_applied"] is False
                dialog.stage()
                dialog.pool.shutdown(wait=True)
                assert block.parameters == original
                assert inspector.apply()
                inspector.open_plugin_dialog()
                dialog = inspector._plugin_dialog
                np.testing.assert_array_equal(dialog.session.recipe.mesh.triangles, triangles)
                np.testing.assert_array_equal(dialog.session.recipe.part.points, points)
                assert reference_capture(dialog.session.recipe).capture_id == capture.capture_id
                dialog.preview_reference()
                settle(app, dialog)
                assert len(dialog.session.result.picks) == expected
                dialog.reject()
                dialog.pool.shutdown(wait=True)
                session.save(destination)
            finally:
                window.dirty = False
                window.close()
                app.processEvents()
        with ProjectSession.open(destination, registry) as reopened:
            reopened.run()
            block = reopened.graph.blocks["jaw"]
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["count"] == expected
            report = result.outputs["report"]
            write_json(folder / "expected.json", report)
            for key in ("graphics", "mesh_fit"):
                cloud = result.outputs[key]
                assert cloud.colors is not None
                write_cloud(folder / f"{key}.ply", cloud, binary=True)
                write_json(folder / f"{key}.json", cloud.metadata)
                assert cv2.imwrite(
                    str(folder / f"{key}.png"),
                    render_cloud(
                        cloud, 1000, 740, yaw=155, pitch=-35, limit=40_000, stable_scale=True
                    ),
                )
            (folder / "result.png").write_bytes((folder / "graphics.png").read_bytes())
            rows.append(
                dict(
                    case=name,
                    eligible=expected,
                    triangles=len(triangles),
                    original_part_frame=old.part.frame,
                    original_origin_preserved=True,
                    actual_triangle_file_import=True,
                    cancel_preserved=True,
                    connected_reference_retained=True,
                    stage_did_not_apply=True,
                    inspector_apply_reopen=True,
                    independent_reference_preview=True,
                    mesh_acceptance_applied=False,
                    source_cad_sha256=hashlib.sha256(cad.read_bytes()).hexdigest(),
                    expected_sha256=hashlib.sha256(
                        (folder / "expected.json").read_bytes()
                    ).hexdigest(),
                    assistance="Authored pose"
                    if name == "clear"
                    else "T-LESS annotated object pose",
                )
            )
    write_json(output / "checks.json", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    initial = builtin_registry()
    load_plugin(args.root.resolve() / "plugin.py", initial)
    build(args.root.resolve(), args.output.resolve())
