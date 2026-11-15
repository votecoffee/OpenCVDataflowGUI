import copy
import hashlib
import math
import json
from xml.etree import ElementTree as ET

import pytest
from opencv_bin_picking_fixture.scene import default_scene, validate, world_xml
from opencv_bin_picking_fixture.state import FixtureState, Observation
from opencv_bin_picking_fixture.run import environment
from opencv_bin_picking_fixture.scene import digest


def mesh_scene():
    scene = default_scene()
    for part in scene["parts"]:
        part["size"] = [0.04, 0.04, 0.046]
        part["mesh"] = {kind: "assets/part.stl" for kind in ("visual", "collision")}
        for kind in ("visual", "collision"):
            part["mesh"][kind + "_sha256"] = hashlib.sha256(b"mesh").hexdigest()
        part["axial_grasp"] = {
            "yaw_symmetry": "continuous_z",
            "jaw_width_m": 0.04,
            "tcp_above_center_m": 0.018,
        }
    return scene


def test_mesh_world_uses_separate_hashed_geometry_with_explicit_unit_scale(tmp_path):
    scene = mesh_scene()
    root = ET.fromstring(world_xml(scene, tmp_path))
    part = root.find("world/model[@name='part_01']/link")
    for kind in ("visual", "collision"):
        assert part.find(kind + "/geometry/box") is None
        uri = part.findtext(kind + "/geometry/mesh/uri")
        expected = "file://" + (tmp_path / "assets/part.stl").as_posix()
        assert (
            uri == expected
            if tmp_path.drive
            else uri == (tmp_path / "assets/part.stl").as_uri()
        )
        assert part.findtext(kind + "/geometry/mesh/scale") == "1 1 1"


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: p["mesh"].__setitem__("visual", "../outside.stl"),
        lambda p: p["mesh"].__setitem__("collision_sha256", "unknown"),
        lambda p: p["axial_grasp"].__setitem__("jaw_width_m", float("nan")),
    ],
)
def test_mesh_and_authored_grasp_metadata_fail_closed(edit):
    scene = mesh_scene()
    edit(scene["parts"][0])
    with pytest.raises(ValueError):
        validate(scene)


def test_round_grasp_accepts_symmetry_but_keeps_height_jaw_and_freshness_gates():
    scene = mesh_scene()
    clock = [100.0]
    state = FixtureState(scene, now=lambda: clock[0])
    part = scene["parts"][0]
    state.poses[part["id"]] = Observation(
        tuple(part["pose"][:3]), (0, 0, math.sin(0.6), math.cos(0.6)), 100, 1
    )
    state.tcp = Observation(
        (0.45, -0.12, part["pose"][2] + 0.018), (1, 0, 0, 0), 100, 1
    )
    state.joints = {
        n: 0.02 for n in ("gripper_left_finger_joint", "gripper_right_finger_joint")
    }
    state.joints_received = 100
    state.arm_stationary = True
    state.camera_ready = True
    state.gate(part["id"])
    for edit in (
        lambda s: setattr(s, "joints_received", 90),
        lambda s: s.joints.__setitem__("gripper_left_finger_joint", 0.03),
        lambda s: setattr(
            s, "tcp", Observation((0.45, -0.12, 0.35), (1, 0, 0, 0), 100, 1)
        ),
    ):
        bad = copy.deepcopy(state)
        edit(bad)
        with pytest.raises(ValueError):
            bad.gate(part["id"])
    del part["axial_grasp"]
    with pytest.raises(ValueError, match="40 mm"):
        state.gate(part["id"])


def test_launch_rechecks_scene_and_mesh_after_preparation(tmp_path):
    scene=mesh_scene()
    (tmp_path/'assets').mkdir()
    asset=tmp_path/'assets/part.stl';asset.write_bytes(b'mesh')
    (tmp_path/'scene.json').write_text(json.dumps(scene),encoding='utf-8')
    (tmp_path/'preparation.json').write_text(json.dumps({
        'overlay':str(tmp_path),'runtime_root':'native','scene_sha256':digest(scene),
    }),encoding='utf-8')
    environment(tmp_path,'opencv_bin_pick_hash_test',current={})
    asset.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Mesh changed'):
        environment(tmp_path,'opencv_bin_pick_hash_test',current={})
    asset.write_bytes(b'mesh')
    scene['parts'][0]['mass']=.051
    (tmp_path/'scene.json').write_text(json.dumps(scene),encoding='utf-8')
    with pytest.raises(ValueError,match='Scene changed'):
        environment(tmp_path,'opencv_bin_pick_hash_test',current={})
