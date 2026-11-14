import math

import numpy as np
import pytest

from opencv_dataflow_pick_backend.kinematics import SerialChain, joint_samples

URDF = """<robot name="test">
<joint name="a" type="revolute"><parent link="world"/><child link="arm"/>
<origin xyz="1 2 3"/><axis xyz="0 0 1"/></joint>
<joint name="b" type="prismatic"><parent link="arm"/><child link="slide"/>
<origin xyz="2 0 0"/><axis xyz="1 0 0"/></joint>
<joint name="tool" type="fixed"><parent link="slide"/><child link="tcp"/>
<origin xyz="0 0 0.5" rpy="0 0 0.2"/></joint></robot>"""


def test_analytic_rotation_translation_and_fixed_tcp():
    model = SerialChain(URDF, tip="tcp")
    pose, skeleton = model.forward({"a": math.pi / 2, "b": 0.25})
    np.testing.assert_allclose(pose[:3, 3], [1, 4.25, 3.5], atol=1e-12)
    np.testing.assert_allclose(
        pose[:2, :2],
        [[-math.sin(0.2), -math.cos(0.2)], [math.cos(0.2), -math.sin(0.2)]],
        atol=1e-12,
    )
    assert len(skeleton) == 4 and model.required == ("a", "b")
    assert (
        SerialChain(URDF.replace('name="test"', 'name="other"'), tip="tcp").sha256 == model.sha256
    )
    copied = model.record()
    copied["joints"].clear()
    assert len(model.record()["joints"]) == 3


def test_trajectory_keeps_endpoints_and_rejects_invalid_skipped_states():
    model = SerialChain(URDF, tip="tcp")
    points = [dict(positions=[i * 0.1], time_from_start=dict(sec=i, nanosec=0)) for i in range(10)]
    trace = model.trajectory(dict(joint_names=["a"], points=points), {"b": 0}, maximum=3)
    assert trace["times_s"] == [0, 4, 9] and trace["decimated"]
    expected = np.array([[1 + 2 * math.cos(a), 2 + 2 * math.sin(a), 3.5] for a in (0, 0.4, 0.9)])
    np.testing.assert_allclose(trace["positions_m"], expected, atol=1e-12)
    points[2]["positions"] = [float("nan")]
    with pytest.raises(ValueError, match="invalid"):
        model.trajectory(dict(joint_names=["a"], points=points), {"b": 0}, maximum=3)


def test_bad_chains_and_missing_joints_remain_errors():
    with pytest.raises(ValueError, match="chain"):
        SerialChain(URDF, root="missing", tip="tcp")
    with pytest.raises(ValueError, match="entities"):
        SerialChain("<!DOCTYPE robot>" + URDF, tip="tcp")
    with pytest.raises(KeyError):
        SerialChain(URDF, tip="tcp").forward({"a": 0})


def test_equivalent_joint_windings_preserve_pose_and_respect_limits():
    text = URDF.replace(
        '<axis xyz="0 0 1"/>',
        '<axis xyz="0 0 1"/><limit lower="-6.28318530718" upper="6.28318530718"/>',
    )
    model = SerialChain(text, tip="tcp")
    candidates = model.joint_candidates({"a": -3.1, "b": 0.1}, {"a": 0, "b": 0})
    assert any(c["a"] > 3 for c in candidates)
    expected, _ = model.forward(candidates[0])
    for value in candidates:
        actual, _ = model.forward(value)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert -2 * math.pi - 1e-9 <= value["a"] <= 2 * math.pi + 1e-9
        assert value["b"] == 0.1


def test_post_retiming_samples_bound_changes_and_preserve_other_joints():
    result = joint_samples(
        dict(joint_names=["a"], points=[dict(positions=[0.035])]), {"a": 0, "finger": 0.02}
    )
    assert len(result) == 4 and all(r["finger"] == 0.02 for r in result)
    np.testing.assert_allclose([r["a"] for r in result], [0.00875, 0.0175, 0.02625, 0.035])
    with pytest.raises(ValueError, match="limit"):
        joint_samples(
            dict(joint_names=["a"], points=[dict(positions=[10])]), {"a": 0}, maximum_states=10
        )
