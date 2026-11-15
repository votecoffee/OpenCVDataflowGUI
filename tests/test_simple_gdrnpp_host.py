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
