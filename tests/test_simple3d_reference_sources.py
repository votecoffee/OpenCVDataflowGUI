import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import read_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


@pytest.mark.local_examples
@pytest.mark.parametrize("folder", ("reference_sources", "boundary_verification"))
def test_real_reference_sources_keep_hashes_native_fit_and_known_false_positives(folder):
    pytest.importorskip("opencv_simple_3d_locate")
    from opencv_simple_3d_locate.optimization import pose_agreement

    root = (Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate"
            / folder)
    for name, digest in json.loads((root / "artifact_hashes.json").read_bytes()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for name, digest in json.loads((root / "source_hashes.json").read_bytes()).items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    summary = json.loads((root / "summary.json").read_bytes())
    assert len(summary["cases"]) == 66 and summary["repeats"] == 2
    assert all(row["repeatable_matches"] for row in summary["cases"])
    assert summary["teaching_captures_excluded_from_test_totals"]
    assert not summary["runtime_pose_seeds"] and not summary["gate_settings_tuned"]
    assert summary["sources"] == {
        "cad": {"matched": 0, "missed": 16, "extra": 0, "negative_extra": 0},
        "reconstructed": {"matched": 2, "missed": 14, "extra": 18, "negative_extra": 7},
        "captured": {"matched": 2, "missed": 14, "extra": 25, "negative_extra": 10},
    }
    assert len(summary["projects"]) == 7
    if folder == "boundary_verification":
        assert summary["boundary_verification"] == "auto"
        assert any(json.loads((root / row["report"]).read_bytes())["runs"][0]["result"]
                   ["execution"]["rejection_counts"].get("boundary", 0)
                   for row in summary["cases"])
    metadata = json.loads((root / "source/models_cad/models_info.json").read_bytes())
    for item in summary["projects"]:
        path = root / item["project"]
        graph = Graph.load(path, builtin_registry())
        try:
            graph.run()
            block = graph.blocks["locate"]
            assert not block.error and block.preview_output == "fit"
            assert block.parameters["render_overlay"] is False
            outputs = block.last_result.outputs
            fit = outputs["fit"]
            assert isinstance(fit, PointCloud) and fit.frame == "camera" and fit.units == "mm"
            assert fit.colors.shape == fit.points.shape
            saved = read_cloud(root / path.name.replace("project_", "result_")
                               .replace(".json", ".ply"), units="mm", frame="camera")
            np.testing.assert_array_equal(fit.points, saved.points)
            np.testing.assert_array_equal(fit.colors, saved.colors)
            assert outputs["count"] == item["accepted_count"]
            # Recipe filenames identify the canonical object independently of scene inputs.
            object_id = int(Path(block.parameters["recipe_path"]).stem.split("object")[1])
            scene_name = Path(graph.blocks["scene"].parameters["path"]).stem
            scene, frame, _ = scene_name.split("_", 2)
            entries = json.loads((root / "source/test_primesense" / scene[5:]
                                  / "scene_gt.json").read_bytes())[str(int(frame[5:]))]
            truth = []
            for entry in entries:
                if entry["obj_id"] == object_id:
                    pose = np.eye(4)
                    pose[:3, :3] = np.asarray(entry["cam_R_m2c"]).reshape(3, 3)
                    pose[:3, 3] = entry["cam_t_m2c"]
                    truth.append(pose)
            symmetries = [np.asarray(s).reshape(4, 4)
                          for s in metadata[str(object_id)].get("symmetries_discrete", [])]
            actual = pose_agreement(
                [np.asarray(m["source_to_target"]) for m in outputs["report"]["matches"]],
                truth, position_tolerance=5, angle_degrees=5, symmetries=symmetries,
            )
            assert {k: actual[k] for k in ("matched", "missed", "extra")} == item["expected"]
            if item["outcome"] == "false_positive":
                assert not truth and outputs["count"] == actual["extra"] == 5
        finally:
            graph.close()
