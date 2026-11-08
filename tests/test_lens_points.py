"""Cross-check sparse undistortion against the image maps and their frame contract."""

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock

_PROFILE = {
    "model": "opencv_pinhole_v1",
    "camera_matrix": [[430, 0, 319.5], [0, 425, 239.5], [0, 0, 1]],
    "distortion": [-0.25, 0.09, 0.001, -0.001, 0],
    "image_size": [640, 480],
    "units": "mm",
    "rms_pixels": 0.07,
}


def _run(block, inputs):
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    return result


def test_lens_points_share_the_corrected_image_frame_and_keep_order(tmp_path):
    registry = builtin_registry()
    builder = registry.create(
        "calibration.lens_maps",
        parameters={"width": 480, "height": 360, "rotation_vector": [0.04, -0.08, 0.01]},
    )
    maps = _run(builder, {"calibration": _PROFILE})

    # Reload the map archive to exercise the persisted correction identity.
    path = str(tmp_path / "maps.npz")
    wire_maps = {name: packet_for(maps, name) for name in ("map_x", "map_y")}
    _run(registry.create("output.lens_maps", parameters={"path": path}), wire_maps)
    maps = _run(registry.create("input.lens_maps", parameters={"path": path}), {})
    rectified = maps.outputs["rectified_calibration"]

    camera = np.asarray(_PROFILE["camera_matrix"], np.float64)
    distortion = np.asarray(_PROFILE["distortion"])
    rays = np.array([[0.4, 0.15, 1], [-0.44, -0.27, 1], [0, 0, 1]])
    zeros = np.zeros(3)
    raw = cv2.projectPoints(rays, zeros, zeros, camera, distortion)[0].reshape(-1, 2)
    payload = {
        "spatial": frame([640, 480], "raw-camera"),
        "ticket": TicketClock("test").issue("camera"),
    }
    block = registry.create("calibration.undistort_lens_points")
    block.exposed_outputs.append(RESULT_GRAPHICS)
    result = _run(
        block,
        {
            "points": ValuePacket(raw.tolist(), payload),
            "calibration": _PROFILE,
            "rectified_calibration": packet_for(maps, "rectified_calibration"),
        },
    )
    expected = rays @ np.asarray(rectified["rectification_rotation"]).T
    expected = expected @ np.asarray(rectified["camera_matrix"]).T
    np.testing.assert_allclose(
        result.outputs["points"], expected[:, :2] / expected[:, 2, None], atol=1e-5
    )
    assert result.outputs["inside"] == [True] * 3
    assert result.outputs["roundtrip_max_error"] < 1e-5
    corrected_image = _run(
        registry.create("cv.remap"),
        {
            "image": ValuePacket(np.zeros((480, 640), np.uint8), payload),
            **{name: packet_for(maps, name) for name in ("map_x", "map_y")},
        },
    )
    context = result.payloads["points"]
    assert context["spatial"] == corrected_image.payloads["image"]["spatial"]
    assert context["spatial"] != payload["spatial"]
    assert context["ticket"] == payload["ticket"]
    (layer,) = result.outputs[RESULT_GRAPHICS].layers
    assert layer.payload["spatial"] == context["spatial"]
    assert len(layer.objects) == 3


def test_lens_points_reject_mixed_profiles_and_withhold_inaccurate_points():
    registry = builtin_registry()
    maps = _run(registry.create("calibration.lens_maps"), {"calibration": _PROFILE})
    block = registry.create("calibration.undistort_lens_points")
    inputs = {
        "points": [[25, 30]],
        "calibration": {**_PROFILE, "distortion": [0.0] * 5},
        "rectified_calibration": packet_for(maps, "rectified_calibration"),
    }
    assert block.execute(inputs) is None
    assert "must come from maps" in block.error

    inputs["calibration"] = _PROFILE
    changed_grid = frame([640, 480], "shifted")
    changed_grid["to_root"][0][2] = 3.0
    inputs["points"] = ValuePacket([[25, 30]], {"spatial": changed_grid})
    assert block.execute(inputs) is None
    assert "original calibrated pixel grid" in block.error

    inputs["points"] = [[25, 30]]
    block.parameters["max_iterations"] = 1
    result = block.execute(inputs)
    assert result is not None and not result.passed, block.error
    assert result.outputs["points"] is NO_VALUE
    assert result.outputs["roundtrip_max_error"] > 0.05
    assert result.fail_reasons[0]["code"] == "lens_points_roundtrip_above_limit"
