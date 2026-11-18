"""Replay published projects, portable Save As, and optional native cloud review."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin


def verify(root, output, *, native=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    app = None
    if native:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    rows = []
    for path in sorted(root.rglob("project.json")):
        expected = json.loads(path.with_name("expected.json").read_bytes())
        relative = path.parent.relative_to(root).as_posix()
        with ProjectSession.open(path, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.run()
            block = session.graph.blocks["jaw"]
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["report"] == expected
            assert result.passed is (expected["eligible_grasps"] > 0)
            cloud = result.outputs["graphics"]
            pose_fit = result.outputs["pose_fit"]
            mesh_fit = result.outputs["mesh_fit"]
            assert cloud.colors is not None and cloud.metadata["derived_review"]
            assert cloud.units == "m" and cloud.frame == expected["frame"]
            if not result.passed:
                assert result.outputs["tcp"] is NO_VALUE
                assert result.outputs["pregrasp"] is NO_VALUE
            if app is not None:
                from opencv_dataflow.app import MainWindow

                window = MainWindow()
                try:
                    load_plugin(root / "plugin.py", window.registry)
                    window.replace_graph(session.graph, path)

                    # Native review must consume the saved result without re-planning.
                    def forbidden(*args, **kwargs):
                        raise AssertionError("Native result review re-ran the jaw planner")

                    block._planner.plan = forbidden
                    if "locate" in session.graph.blocks:
                        session.graph.blocks["locate"]._locator.locate = forbidden
                    window.show()
                    app.processEvents()
                    node = window.scene.nodes["jaw"]
                    assert node.thumbnail_value() is cloud
                    before = node._cloud_yaw
                    window.view.animate_cloud_thumbnails()
                    assert node._cloud_yaw != before
                    window.inspector.select(block)
                    window.inspector.tabs.setCurrentWidget(window.inspector.results_page)
                    app.processEvents()
                    assert window.inspector.preview_choice.currentText() == "Cloud / graphics"
                    window.inspector.open_image_popout()
                    (viewer,) = window.inspector.image_popouts.values()
                    assert viewer.cloud is cloud and viewer.output_name == "graphics"
                    assert not viewer.image.pixmap().isNull()
                    viewer.yaw.setValue(155)
                    viewer.flyaround.setChecked(True)
                    assert viewer.orbit_timer.isActive()
                    yaw = viewer.yaw.value()
                    viewer.advance_orbit()
                    assert viewer.yaw.value() != yaw
                    np.testing.assert_array_equal(viewer.cloud.colors, cloud.colors)
                    viewer.close()
                    assert not viewer.orbit_timer.isActive()
                    for name, review_cloud in (("pose_fit", pose_fit), ("mesh_fit", mesh_fit)):
                        if review_cloud is NO_VALUE:
                            continue
                        block.preview_output = name
                        assert node.thumbnail_value() is review_cloud
                        window.inspector.preview_choice.setCurrentText(f"Cloud / {name}")
                        window.inspector.open_image_popout()
                        viewer = next(
                            v
                            for v in window.inspector.image_popouts.values()
                            if v.output_name == name
                        )
                        assert viewer.cloud is review_cloud
                        np.testing.assert_array_equal(viewer.cloud.colors, review_cloud.colors)
                        viewer.flyaround.setChecked(True)
                        yaw = viewer.yaw.value()
                        viewer.advance_orbit()
                        assert viewer.yaw.value() != yaw
                        assert not viewer.image.pixmap().isNull()
                        viewer.close()
                        block.preview_output = "graphics"
                finally:
                    window.dirty = False
                    window.close()
                    app.processEvents()
            # Saving a copy retains all numerical assets and the immutable recipe.
            saved = output / relative / "project.json"
            saved.parent.mkdir(parents=True)
            session.save(saved)
        with ProjectSession.open(saved, registry) as copied:
            copied.run()
            block = copied.graph.blocks["jaw"]
            assert block.last_result is not None, block.error
            assert block.last_result.outputs["report"] == expected
            asset_paths = [
                "assets/depth.npy",
                "assets/records.json",
                block.parameters["recipe_path"],
            ]
            if "locate" in copied.graph.blocks:
                asset_paths += ["assets/full.ply", "assets/isolated.ply", "assets/locator.s3l"]
                for name in ("locator_scene_path", "locator_depth_path"):
                    assert copied.graph.blocks["capture"].parameters[name].startswith("assets/")
            for relative_asset in asset_paths:
                assert (saved.parent / relative_asset).read_bytes() == (
                    path.parent / relative_asset
                ).read_bytes()
        rows.append(
            dict(
                case=relative,
                passed=expected["eligible_grasps"] > 0,
                eligible=expected["eligible_grasps"],
                reopened_exact=True,
                portable_save_as_exact=True,
                native_flyaround_colors_preserved=bool(native),
                review_did_not_replan=bool(native),
                native_pose_fit_review=bool(native and pose_fit is not NO_VALUE),
                native_mesh_fit_review=bool(native and mesh_fit is not NO_VALUE),
                pose_fit_instances=(
                    len(pose_fit.metadata["pose_fit"]) if pose_fit is not NO_VALUE else 0
                ),
                expected_sha256=hashlib.sha256(
                    path.with_name("expected.json").read_bytes()
                ).hexdigest(),
            )
        )
    assert len(rows) in (5, 8, 10, 12, 14)
    (output / "checks.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    verify(args.root.resolve(), args.output.resolve(), native=args.native)
