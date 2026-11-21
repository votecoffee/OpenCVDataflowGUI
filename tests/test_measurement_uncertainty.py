"""Validate planar uncertainty against closed-form and repeated synthetic controls."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.measurement_uncertainty import MeasurePlanarUncertainty
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/measurement_uncertainty"


def _rotation(rvec):
    x, y, z = np.asarray(rvec).T
    zero = np.zeros_like(x)
    skew = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape((-1, 3, 3))
    theta = np.linalg.norm(rvec, axis=-1).reshape(-1, 1, 1)
    return (
        np.eye(3)
        + np.sinc(theta / np.pi) * skew
        + 0.5 * np.sinc(theta / (2 * np.pi)) ** 2 * (skew @ skew)
    )


def _brown_offsets(x, y, k1, k2, p1, p2):
    squared = x**2 + y**2
    radial = 1 + squared * (k1 + squared * k2)
    return (
        radial,
        2 * p1 * x * y + p2 * (squared + 2 * x**2),
        p1 * (squared + 2 * y**2) + 2 * p2 * x * y,
    )


def _repeated_distances(pose, camera, points, report, joint, seed, trials=24_000):
    assert camera["distortion"][4] == 0
    r0 = np.array(pose["rvec"])
    t0 = np.array(pose["translation"])
    target = points @ _rotation(r0)[0].T + t0
    x, y = (target[:, :2] / target[:, 2:3]).T
    k1, k2, p1, p2 = camera["distortion"][:4]
    fx, fy, cx, cy = (camera[key] for key in ("fx", "fy", "cx", "cy"))
    radial, dx, dy = _brown_offsets(x, y, k1, k2, p1, p2)
    px = x * radial * fx + dx * fx + cx
    py = y * radial * fy + dy * fy + cy

    rng = np.random.default_rng(seed)
    variables = dict(
        zip(
            joint["terms"],
            rng.multivariate_normal(np.zeros(len(joint["terms"])), joint["matrix"], trials).T,
            strict=True,
        )
    )
    for key in report["independent_pixel_terms"]:
        variables[key] = rng.normal(0, report["pixel_stddev_px"], trials)

    x_d = (
        px + np.column_stack((variables["u0"], variables["u1"])) - (cx + variables["cx"][:, None])
    ) / (fx + variables["fx"][:, None])
    y_d = (py + np.column_stack((variables["v0"], variables["v1"])) - cy) / fy
    x, y = x_d.copy(), y_d.copy()
    for _ in range(15):
        radial, dx, dy = _brown_offsets(x, y, k1 + variables["k1"][:, None], k2, p1, p2)
        x, y = (x_d - dx) / radial, (y_d - dy) / radial
    rays = np.stack((x, y, np.ones_like(x)), axis=2)
    rotation = _rotation(r0 + np.column_stack((variables["rx"], np.zeros((trials, 2)))))
    translation = t0 + np.column_stack((np.zeros((trials, 2)), variables["tz"]))
    normal = rotation[:, :, 2]
    denominator = (rays @ normal[:, :, None]).squeeze(2)
    depth = np.sum(normal * translation, axis=1, keepdims=True) / denominator
    xy = (rays * depth[:, :, None] - translation[:, None, :]) @ rotation[:, :, :2]
    return np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1), xy.reshape(-1, 4)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_project_matches_repeated_correlated_measurements(profile):
    graph = Graph.load(EXAMPLE / "measurement_uncertainty.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["measure"]
        result = block.last_result
        assert result and result.passed, block.error
        report = result.outputs["report"]
        pose, camera = (graph.blocks[key].parameters for key in ("plane", "camera"))
        expected = np.array(graph.blocks["segment"].parameters["points"])
        length = np.linalg.norm(expected[1] - expected[0])
        np.testing.assert_allclose(report["plane_points_xy"], expected[:, :2], atol=1e-10, rtol=0)
        assert report["nominal_distance"] == pytest.approx(length, abs=1e-10)
        assert report["units"] == "mm" and report["plane_frame"] == "board"
        assert bool(result.drawings) == (profile == "debug")
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["reference"].last_result.outputs["image"]
        )

        distances, endpoints = _repeated_distances(
            pose,
            camera,
            expected,
            report,
            block.parameters["input_covariance"],
            seed=87122,
        )
        observed = distances.std(ddof=1)
        assert observed == pytest.approx(report["standard_uncertainty"], rel=0.025)
        low, high = report["interval"]
        assert 0.943 < np.mean((low <= distances) & (distances <= high)) < 0.958
        observed_xy_cov = np.cov(endpoints.T)
        error = np.linalg.norm(observed_xy_cov - report["endpoint_covariance_xy"])
        assert error < 0.035 * np.linalg.norm(observed_xy_cov)
    finally:
        graph.close()


@pytest.fixture
def front():
    registry = builtin_registry()
    camera = registry.create("3d.camera", parameters={"units": "mm"}).execute({}).outputs["camera"]
    transform = (
        registry.create(
            "3d.pose",
            parameters={"translation": [0, 0, 500], "units": "mm"},
        )
        .execute({})
        .outputs["transform"]
    )
    return {
        "camera": camera,
        "plane_to_camera": transform,
        "image_points": [[210, 240], [390, 240]],
    }


@pytest.mark.parametrize("corr, expected", ((1.0, 0.0), (0.0, 0.5 * np.sqrt(2)), (-1.0, 1.0)))
def test_common_and_opposing_pixel_errors_use_joint_covariance(front, corr, expected):
    raw = dict(terms=["u1", "u0"], matrix=[[0.25, 0.25 * corr], [0.25 * corr, 0.25]])
    block = MeasurePlanarUncertainty()
    result = block.execute({**front, "covariance": raw})
    assert result and result.passed, block.error
    assert result.outputs["distance"] == pytest.approx(180)
    assert result.outputs["standard_uncertainty"] == pytest.approx(expected, abs=1e-12)
    assert result.outputs["report"]["joint_covariance_terms"] == ["u1", "u0"]


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"terms": ["u1", "u0"], "matrix": [[1, 2], [2, 1]]}, "positive semidefinite"),
        ({"terms": ["u1", "u0"], "matrix": [[1, 0.25], [0.5, 1]]}, "symmetric"),
        ({"terms": ["u1", "u0"], "matrix": [[1, 0.2], [0.2, 0]]}, "Zero-variance"),
        ({"terms": ["u1", "u1"], "matrix": [[1, 0], [0, 1]]}, "distinct"),
        ({"terms": ["k1"], "matrix": [[0.03]]}, "present in the camera"),
        ({"terms": ["u1"], "matrix": [[0]]}, "Declare nonzero"),
        ({"terms": ["u1"], "matrix": [["0.1"]]}, "numeric matrix"),
    ],
)
def test_malformed_uncertainty_is_explicit(front, raw, error):
    block = MeasurePlanarUncertainty()
    assert block.execute({**front, "covariance": raw}) is None
    assert error in block.error


def test_geometry_failure_and_tolerance_limit(front):
    block = MeasurePlanarUncertainty(
        parameters={
            "pixel_stddev_px": 0.5,
            "maximum_expanded_uncertainty": 0.15,
        }
    )
    too_wide = block.execute(front)
    assert too_wide and not too_wide.passed
    assert too_wide.fail_reasons[0]["code"] == "measurement_uncertainty_limit"
    assert too_wide.outputs["distance"] == pytest.approx(180)

    close = block.execute({**front, "image_points": [[320, 240], [320.2, 240]]})
    assert close and not close.passed and close.outputs["distance"] is NO_VALUE
    assert close.fail_reasons[0]["code"] == "measurement_linearization"

    registry = builtin_registry()
    pose = (
        registry.create(
            "3d.pose",
            parameters={"translation": [0, 0, 500], "units": "mm", "rvec": [0, np.pi / 2, 0]},
        )
        .execute({})
        .outputs["transform"]
    )
    horizon = block.execute(
        {**front, "plane_to_camera": pose, "image_points": [[320, 200], [320, 230]]}
    )
    assert horizon and not horizon.passed
    assert horizon.fail_reasons[0]["code"] == "measurement_horizon"
    assert horizon.outputs["standard_uncertainty"] is NO_VALUE


def test_known_grid_and_pose_must_match(front):
    block = MeasurePlanarUncertainty(parameters={"pixel_stddev_px": 0.4})
    wrong = {**front["plane_to_camera"], "units": "m"}
    assert block.execute({**front, "plane_to_camera": wrong}) is None
    assert "length units" in block.error
    payload = {"spatial": frame([320, 240], "thumbnail")}
    tagged = ValuePacket(front["image_points"], payload)
    assert block.execute({**front, "image_points": tagged}) is None
    assert "grid does not match" in block.error


def test_points_preserve_capture_without_an_image_and_reject_stale_overlay(front):
    block = MeasurePlanarUncertainty(parameters={"pixel_stddev_px": 0.4})
    clock = TicketClock("measurement-test")
    payload = {"spatial": frame([640, 480], "camera-image"), "ticket": clock.issue("camera")}
    tagged = ValuePacket(front["image_points"], payload)
    inputs = {**front, "image_points": tagged}
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.payloads["distance"]["ticket"] == payload["ticket"]
    assert result.payloads["report"]["spatial"] == payload["spatial"]

    next_payload = {**payload, "ticket": clock.issue("camera")}
    stale = ValuePacket(np.zeros((480, 640), np.uint8), next_payload)
    assert block.execute({**inputs, "image": stale}) is None
    assert "same capture" in block.error
