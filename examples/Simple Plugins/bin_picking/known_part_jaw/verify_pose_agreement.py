"""Verify frozen study clouds in native fly-around views; file replay only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry


def verify(root, output, *, native=False):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    app = None
    if native:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    results = []
    for obj in (6, 8):
        folder = root / f"review_object{obj}"
        views = json.loads((folder / "views.json").read_bytes())
        expected = {}
        with ProjectSession.open(folder / "review.json", registry) as session:
            assert session.inspect()["validation"]["valid"]
            assert all(block.type_id == "input.cloud" for block in session.graph.blocks.values())
            session.run()
            window = None
            if app is not None:
                from opencv_dataflow.app import MainWindow

                window = MainWindow()
                window.replace_graph(session.graph, folder / "review.json")
                window.show()
                app.processEvents()
            try:
                for view in views:
                    path = folder / view["path"]
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == view["sha256"]
                    block = session.graph.blocks[view["block"]]
                    assert block.last_result is not None, block.error
                    cloud = block.last_result.outputs["cloud"]
                    assert cloud.units == "m" and cloud.frame == view.get("frame", "study_camera")
                    assert cloud.colors is not None and len(cloud.points) > 0
                    expected[block.id] = cloud
                    if window is not None:
                        node = window.scene.nodes[block.id]
                        assert node.thumbnail_value() is cloud
                        yaw = node._cloud_yaw
                        window.view.animate_cloud_thumbnails()
                        assert node._cloud_yaw != yaw
                        inspector = window.inspector
                        inspector.select(block)
                        inspector.tabs.setCurrentWidget(inspector.results_page)
                        app.processEvents()
                        assert inspector.preview_choice.currentText() == "Cloud / cloud"
                        inspector.open_image_popout()
                        viewer = next(
                            v for v in inspector.image_popouts.values() if v.cloud is cloud
                        )
                        np.testing.assert_array_equal(viewer.cloud.colors, cloud.colors)
                        assert not viewer.image.pixmap().isNull()
                        viewer.flyaround.setChecked(True)
                        assert viewer.orbit_timer.isActive()
                        yaw = viewer.yaw.value()
                        viewer.advance_orbit()
                        assert viewer.yaw.value() != yaw
                        viewer.close()
                        assert not viewer.orbit_timer.isActive()
            finally:
                if window is not None:
                    window.dirty = False
                    window.close()
                    app.processEvents()
            saved = output / f"review_object{obj}" / "review.json"
            saved.parent.mkdir()
            session.save(saved)
        # Built-in file inputs rebase to the original assets when saving elsewhere.
        with ProjectSession.open(saved, registry) as reopened:
            assert reopened.inspect()["validation"]["valid"]
            reopened.run()
            for key, cloud in expected.items():
                block = reopened.graph.blocks[key]
                assert block.last_result is not None, block.error
                actual = block.last_result.outputs["cloud"]
                np.testing.assert_array_equal(actual.points, cloud.points)
                np.testing.assert_array_equal(actual.colors, cloud.colors)
        results.append(
            dict(
                object_id=obj,
                derived_clouds=len(expected),
                source_hashes_valid=True,
                exact_saved_replay=True,
                save_as_original_asset_references_valid=True,
                native_thumbnail_popout_orbit=bool(native),
                rgb_preserved=True,
                file_inputs_only=True,
                inference_or_motion_executed=False,
                qualification="Replay success only; no accepted pose or grasp asserted",
            )
        )
    (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    verify(args.root.resolve(), args.output.resolve(), native=args.native)
