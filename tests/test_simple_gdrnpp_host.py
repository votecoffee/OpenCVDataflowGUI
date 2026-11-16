"""Native block metadata, typed RGB-D conversion, missing poses and optional graphics."""

import copy

import numpy as np
import pytest

pytest.importorskip("opencv_dataflow")
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import NO_VALUE, frame
from opencv_dataflow.simple_gdrnpp.dataflow import SimpleGDRNPP, connected_capture


def inputs():
    return dict(
        image=ImageArray(np.full((12, 16, 3), [10, 20, 30], np.uint8), "rgb"),
        depth=DepthArray(np.full((12, 16), 1000), "mm"),
        camera=dict(
            matrix=[[20, 0, 8], [0, 20, 6], [0, 0, 1]], image_size=[16, 12], frame="camera"
        ),
        detections=[],
        capture_id="current-B",
    )


def test_rgb_encoding_physical_depth_and_declared_coordinate_roots():
    values = inputs()
    before = values["image"].copy()
    current, arrays = connected_capture(values, {})
    np.testing.assert_array_equal(arrays["rgb"], values["image"])
    np.testing.assert_allclose(arrays["depth"], 1)
    np.testing.assert_array_equal(values["image"], before)
    assert current["metadata"]["capture_id"] == "current-B"
    with pytest.raises(ValueError, match="coordinate grids"):
        connected_capture(
            values,
            dict(image={"spatial": frame([16, 12], "a")}, depth={"spatial": frame([16, 12], "b")}),
        )
    with pytest.raises(ValueError, match="image_size|image size|intrinsics"):
        connected_capture({**values, "camera": {**values["camera"], "image_size": [12, 16]}}, {})


def test_saved_projects_cannot_choose_an_executable():
    with pytest.raises(ValueError, match="Unknown parameters"):
        SimpleGDRNPP(parameters={"worker_python": "untrusted.py"})


def test_valid_rejection_no_stale_pose_and_graphics_off(tmp_path):
    class Client:
        closed = False

        def call(self, op, data, arrays, **kw):
            self.data = copy.deepcopy(data)
            report = dict(candidates=[], best_pose=None, accepted_candidates=0, capture_id="B")
            return dict(report=report), {}

        def close(self):
            self.closed = True

    block = SimpleGDRNPP(
        parameters=dict(recipe_path="model.gdrnpp", sha256="0" * 64, models_dir=str(tmp_path))
    )
    block.project_dir = tmp_path
    client = Client()
    block._client = client
    block.set_graphics_mode("off")
    result = block.execute(inputs())
    assert result is not None, block.error
    assert result.passed is False and result.outputs["pose"] is NO_VALUE
    assert result.outputs["poses"] == [] and result.outputs["accepted_count"] == 0
    assert result.outputs["overlay"] is NO_VALUE and result.outputs["cloud"] is NO_VALUE
    assert (
        client.data["graphics"] is False
        and client.data["current"]["metadata"]["capture_id"] == "current-B"
    )
    assert block.preview_output == "cloud"
    block.reset_runtime()
    assert client.closed and block._client is None


@pytest.mark.parametrize("suppressed", [False, True])
def test_graphics_follow_best_pose_with_separate_gallery_grid_and_candidate_table(
    tmp_path, suppressed
):
    from opencv_dataflow.simple_gdrnpp.selection import describe_review

    poses = [
        dict(model="pose_3d_v1", source_to_target=np.eye(4).tolist(), identity=i) for i in range(2)
    ]
    report = dict(
        candidates=[
            dict(id="first_rejected", pose=poses[0], fit=dict(status="rejected", coverage=0.99)),
            dict(id="best_accepted", pose=poses[1], fit=dict(status="accepted", coverage=0.8)),
        ],
        best_pose=poses[1],
        accepted_candidates=1,
        capture_id="current-B",
        target_frame="camera",
    )
    if suppressed:
        report.update(overlap_selection={}, depth_accepted_candidates=2)
        report["candidates"][0].update(
            depth_fit=dict(status="accepted", coverage=0.79),
            selection=dict(status="overlap_suppressed", retained_by="best_accepted", overlap=1.0),
        )

    class Client:
        wrong = False

        def call(self, op, data, arrays, **kw):
            selection, _ = describe_review(report, data["review_candidate"])
            return dict(
                report=copy.deepcopy(report),
                candidate_index=0 if self.wrong else selection["candidate_index"],
            ), dict(
                overlay=np.zeros((12, 16, 3), np.uint8),
                gallery=np.zeros((318, 720, 3), np.uint8),
                points=np.zeros((1, 3), np.float32),
                colors=np.zeros((1, 3), np.uint8),
            )

        def close(self):
            pass

    block = SimpleGDRNPP(
        parameters=dict(
            recipe_path="model.gdrnpp",
            sha256="0" * 64,
            models_dir=str(tmp_path),
            render_gallery=True,
        )
    )
    block.project_dir = tmp_path
    block._client = Client()
    result = block.execute(inputs())
    assert result is not None, block.error
    assert result.outputs["pose"] == poses[1] and result.outputs["review"]["candidate_number"] == 2
    assert result.outputs["cloud"].metadata["review"]["candidate_id"] == "best_accepted"
    assert result.outputs["candidate_table"][1]["selected"]
    assert result.outputs["gallery"].shape == (318, 720, 3)
    assert result.payloads["gallery"]["spatial"] != result.payloads.get("overlay", {}).get(
        "spatial"
    )
    block.parameters["review_candidate"] = 1
    rejected = block.execute(inputs())
    assert rejected.outputs["review"]["status"] == (
        "overlap_suppressed" if suppressed else "rejected"
    )
    assert rejected.outputs["accepted_count"] == 1 and rejected.outputs["poses"] == [poses[1]]
    if suppressed:
        assert rejected.outputs["cloud"].metadata["review"]["depth_status"] == "accepted"
        assert "1 retained poses (2 depth-accepted before overlap)" in rejected.message
    assert rejected.passed and rejected.outputs["pose"] == poses[1]
    block.parameters["review_candidate"] = 0
    block._client.wrong = True
    assert block.execute(inputs()) is None and "Update the isolated worker" in block.error
    assert block.last_result is None


@pytest.mark.parametrize(
    "parameters",
    [
        dict(review_candidate=True),
        dict(review_candidate=65),
        dict(review_candidate=-1),
        dict(render_gallery="yes"),
    ],
)
def test_review_controls_validate_without_processing(parameters):
    with pytest.raises(ValueError):
        SimpleGDRNPP(parameters=parameters).validate_parameters()
