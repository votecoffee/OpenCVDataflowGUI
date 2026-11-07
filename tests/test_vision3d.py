"""Real OpenCV 5 3D kernels, typed clouds, file interchange and offscreen viewing."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import read_cloud, write_cloud
from opencv_dataflow.control_flow import byte_size
from opencv_dataflow.core import Graph, Port, validate_value
from opencv_dataflow.examples3d import (
    pose_3d_graph,
    rgbd_cloud_graph,
    stereo_cloud_graph,
    structured_light_graph,
    tsdf_graph,
)
from opencv_dataflow.geometry3d_common import camera_record, pose_matrix, pose_record, rigid
from opencv_dataflow.images import DepthArray, ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.runner import main as run_headless
from opencv_dataflow.toolblocks import encapsulate


def execute(kind, inputs=None, parameters=None, state="pass"):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute(inputs or {})
    assert block.state == state, (kind, block.state, block.error)
    return block.last_result.outputs if block.last_result else {}


def pinhole(w=96, h=72, f=100.0, name="camera", units="m"):
    return camera_record([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], [], (w, h), units, name)


def translation(x=0.0, y=0.0, z=0.0):
    t = np.eye(4)
    t[:3, 3] = [x, y, z]
    return t


def stereo_pair():
    return execute("input.stereo_demo")


EXAMPLES = [
    ("stereo_cloud", stereo_cloud_graph),
    ("rgbd_cloud", rgbd_cloud_graph),
    ("pose_3d", pose_3d_graph),
    ("structured_light", structured_light_graph),
    ("tsdf_fusion", tsdf_graph),
]


@pytest.mark.parametrize("name,factory", EXAMPLES)
@pytest.mark.parametrize("profile", ["debug", "balanced", "lean"])
def test_saved_3d_examples_roundtrip_and_execute(name, factory, profile):
    registry = builtin_registry()
    (project,) = Path("examples").glob(f"*/{name}/{name}.json")
    graph = Graph.load(project, registry)
    assert graph.to_dict() == factory(registry).to_dict()
    graph.set_run_profile(profile)
    graph.run()
    assert all(b.state == "pass" for b in graph.blocks.values()), [
        (b.name, b.error) for b in graph.blocks.values()
    ]
    graph.close()


def test_cloud_is_not_an_image_and_counts_native_storage():
    cloud = PointCloud(np.ones((10, 3)), "mm", "fixture", colors=np.zeros((10, 3), np.uint8))
    validate_value(Port("cloud", "cloud"), cloud)
    assert byte_size(cloud) >= 150
    with pytest.raises(TypeError, match="PointCloud"):
        validate_value(Port("cloud", "cloud"), cloud.points)
    with pytest.raises(TypeError, match="image"):
        validate_value(Port("image"), cloud)
    with pytest.raises(ValueError, match="finite"):
        PointCloud([[0, 0, float("nan")]])
    with pytest.raises(ValueError, match="unit vectors"):
        PointCloud([[0, 0, 0]], normals=[[2, 0, 0]])


def test_depth_projection_units_invalid_samples_and_image_correspondence():
    cam = pinhole(5, 4, 10.0)
    depth = DepthArray(np.full((4, 5), 2000, np.float32), "mm")
    depth[0, 0] = np.nan
    depth[1, 0] = 0
    image = np.zeros((4, 5, 3), np.uint8)
    image[..., 2] = 255
    mask = np.zeros((4, 5), np.uint8)
    mask[:, 0:3] = 255
    result = execute(
        "3d.depth_cloud",
        {"depth": depth, "camera": cam, "image": image, "mask": mask},
        {"sample_step": 1, "output_units": "mm"},
    )
    cloud = result["cloud"]
    assert cloud.units == "mm" and cloud.source_size == (5, 4) and len(cloud.points) == 10
    assert np.all(cloud.colors == [255, 0, 0])
    x, y = cloud.pixels.T
    np.testing.assert_allclose(
        cloud.points, np.c_[200 * (x - 2.5), 200 * (y - 2), np.full(len(x), 2000)], atol=1e-4
    )
    assert [0, 0] not in cloud.pixels.tolist()
    raw = np.full((4, 5), 2000, np.uint16)
    cloud = execute("3d.depth_cloud", {"depth": raw, "camera": cam}, {"sample_step": 1})["cloud"]
    np.testing.assert_allclose(cloud.points[:, 2], 2.0, atol=1e-6)


@pytest.mark.parametrize("algorithm,wls", [("sgbm", False), ("bm", False), ("sgbm", True)])
def test_stereo_disparity_and_q_reprojection_native(algorithm, wls):
    pair = stereo_pair()
    rect = execute("3d.stereo_rectify", {k: pair[k] for k in ("left", "right", "rig")})
    disp = execute(
        "3d.disparity",
        {"left": rect["left"], "right": rect["right"]},
        {"algorithm": algorithm, "wls": wls, "num_disparities": 32},
    )
    assert np.nanmedian(disp["disparity"]) == pytest.approx(8, abs=0.15)
    assert (disp["valid_mask"] == 0).any()
    if wls:
        assert disp["confidence"].dtype == np.float32
    else:
        assert disp["confidence"] is NO_VALUE
    out = execute(
        "3d.reproject",
        {"disparity": disp["disparity"], "rectification": rect["rectification"]},
        {"sample_step": 4},
    )
    assert np.nanmedian(out["depth"]) == pytest.approx(3, abs=0.06)
    assert out["cloud"].frame == "left_camera:rectified" and out["cloud"].units == "m"
    assert out["cloud"].pixels is not None
    assert np.isfinite(out["cloud"].points).all()


def test_calibrated_stereo_estimates_known_baseline_from_varied_views():
    left, right = pinhole(640, 480, 500.0, "left"), pinhole(640, 480, 500.0, "right")
    k = np.asarray(left["camera_matrix"])
    board = np.zeros((35, 3), np.float32)
    board[:, :2] = np.mgrid[:7, :5].T.reshape(-1, 2) * 0.03
    a, b = [], []
    for i in range(6):
        r = np.array([0.06 * i, -0.045 * i, 0.03 * i])
        t = np.array([-0.1 + 0.01 * i, -0.05, 0.65 + 0.07 * i])
        a.append(cv2.projectPoints(board, r, t, k, None)[0].reshape(-1, 2).tolist())
        b.append(cv2.projectPoints(board, r, t + [-0.12, 0, 0], k, None)[0].reshape(-1, 2).tolist())
    result = execute(
        "3d.stereo_calibrate",
        {
            "object_views": [board.tolist()] * 6,
            "left_views": a,
            "right_views": b,
            "left_camera": left,
            "right_camera": right,
        },
    )
    assert result["rms"] < 0.001
    np.testing.assert_allclose(result["rig"]["transform"]["translation"], [-0.12, 0, 0], atol=1e-5)


def test_register_depth_metric_pose_and_invalid_samples():
    a, b = pinhole(40, 30, 50.0, "a"), pinhole(40, 30, 50.0, "b")
    depth = DepthArray(np.full((30, 40), 2000, np.float32), "mm")
    depth[0, 0] = np.nan
    out = execute(
        "3d.register_depth",
        {
            "depth": depth,
            "depth_camera": a,
            "target_camera": b,
            "transform": pose_record(translation(0, 0, 100), "a", "b", "mm"),
        },
    )
    assert out["depth"].units == "m"
    assert float(np.nanmedian(out["depth"])) == pytest.approx(2.1, abs=1e-5)
    execute(
        "3d.register_depth",
        {
            "depth": depth,
            "depth_camera": a,
            "target_camera": b,
            "transform": pose_record(np.eye(4), "wrong", "b"),
        },
        state="error",
    )


@pytest.mark.parametrize("lens", ["pinhole", "fisheye", "omnidir"])
def test_camera_projection_and_rectification_lens_families(lens):
    cam = camera_record(
        [[100, 0, 48], [0, 100, 36], [0, 0, 1]],
        [0] * 4,
        (96, 72),
        lens=lens,
        **({"xi": 0.5} if lens == "omnidir" else {}),
    )
    xyz = [[0, 0, 2], [0.1, -0.2, 2], [0.15, 0.03, 1.5]]
    pose = pose_record(np.eye(4))
    output = execute("3d.project", {"points": xyz, "camera": cam, "transform": pose})
    undistorted = execute("3d.undistort_points", {"points": output["points"], "camera": cam})[
        "points"
    ]
    arr = np.asarray(xyz)
    np.testing.assert_allclose(undistorted, arr[:, :2] / arr[:, 2:3], atol=1e-5)
    image = ImageArray(np.full((72, 96, 3), [200, 0, 0], np.uint8), "rgb")
    out = execute("3d.rectify_camera", {"image": image, "camera": cam}, {"width": 64, "height": 48})
    assert out["image"].shape == (48, 64, 3) and image_format(out["image"]) == "rgb"
    assert out["camera"]["lens"] == "pinhole" and out["camera"]["image_size"] == [64, 48]


@pytest.mark.parametrize(
    "method,ransac,refine",
    [("iterative", True, "lm"), ("epnp", False, "vvs"), ("sqpnp", False, "none")],
)
def test_pnp_recovers_pose_with_projected_correspondences(method, ransac, refine):
    cam = pinhole(640, 480, 500.0)
    xyz = np.random.default_rng(17).uniform([-0.2, -0.1, -0.1], [0.2, 0.1, 0.2], size=(40, 3))
    true = pose_matrix(cv2.Rodrigues(np.array([0.1, -0.2, 0.07]))[0], [0.07, -0.02, 1.6])
    pose = pose_record(true)
    projected = execute("3d.project", {"points": xyz.tolist(), "camera": cam, "transform": pose})
    out = execute(
        "3d.pnp",
        {"object_points": xyz.tolist(), "image_points": projected["points"], "camera": cam},
        {"method": method, "ransac": ransac, "refinement": refine},
    )
    np.testing.assert_allclose(rigid(out["pose"]), true, atol=1e-5)
    assert out["rms"] < 0.001


def test_epipolar_geometry_and_metric_triangulation():
    cam = pinhole(640, 480, 500.0)
    xyz = np.random.default_rng(25).uniform([-0.3, -0.2, 1.0], [0.3, 0.2, 2.0], (48, 3))
    a = execute(
        "3d.project", {"points": xyz.tolist(), "camera": cam, "transform": pose_record(np.eye(4))}
    )
    b = execute(
        "3d.project",
        {
            "points": xyz.tolist(),
            "camera": cam,
            "transform": pose_record(translation(-0.12, 0.01, 0)),
        },
    )
    inputs = {"points_left": a["points"], "points_right": b["points"]}
    f = execute("3d.fundamental", inputs, {"method": "eight_point"})
    error = np.sum(
        (np.c_[b["points"], np.ones(len(xyz))] @ f["matrix"])
        * np.c_[a["points"], np.ones(len(xyz))],
        axis=1,
    )
    assert np.max(np.abs(error)) < 0.01
    rel = execute("3d.essential_pose", {**inputs, "camera": cam})["pose"]
    assert rel["units"] == "unitless" and rel["scale"] == "unknown"
    assert rel["translation"][0] < -0.9
    out = execute(
        "3d.triangulate",
        {**inputs, "projection_left": a["projection"], "projection_right": b["projection"]},
    )
    np.testing.assert_allclose(out["cloud"].points, xyz, atol=1e-5)
    assert all(out["valid"])


def test_cloud_transform_compose_and_reject_mixed_frames():
    first = pose_record(translation(1, 2, 3), "a", "b")
    second = pose_record(translation(-2, 0, 1), "b", "c")
    composed = execute("3d.compose", {"first": first, "second": second})["transform"]
    inverse = execute("3d.compose", {"first": composed}, {"operation": "inverse"})["transform"]
    cloud = PointCloud([[1, 2, 3]], frame="a", normals=[[0, 0, 1]])
    moved = execute("3d.cloud_transform", {"cloud": cloud, "transform": composed})["cloud"]
    assert moved.frame == "c"
    np.testing.assert_allclose(moved.points, [[0, 4, 7]])
    recovered = execute("3d.cloud_transform", {"cloud": moved, "transform": inverse})["cloud"]
    np.testing.assert_allclose(recovered.points, cloud.points)
    execute("3d.cloud_transform", {"cloud": cloud, "transform": second}, state="error")


def test_cloud_filter_uses_original_pixel_frame_and_centroid_drops_pixel_map():
    context = frame([4, 3], "camera-1")
    cloud = PointCloud(
        [[0, 0, 1], [0.1, 0, 1], [1, 0, 1]],
        pixels=[[0, 0], [1, 0], [3, 0]],
        source_size=(4, 3),
        metadata={"spatial": context},
    )
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "rectangle": [0, 0, 2, 1],
        "image_size": [4, 3],
        "coordinate_context": context,
    }
    filtered = execute("3d.cloud_filter", {"cloud": cloud, "region": region})["cloud"]
    assert len(filtered.points) == 2
    execute(
        "3d.cloud_filter",
        {"cloud": cloud, "region": {**region, "coordinate_context": frame([4, 3], "wrong")}},
        state="error",
    )
    sampled = execute(
        "3d.cloud_voxel", {"cloud": filtered}, {"voxel_size": 1.0, "mode": "centroid"}
    )["cloud"]
    assert sampled.pixels is None and sampled.source_size is None
    np.testing.assert_allclose(sampled.points, [[0.05, 0, 1]])


def oriented_surface():
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 24), np.linspace(-0.1, 0.1, 24))
    xyz = np.c_[x.ravel(), y.ravel(), (0.8 + 2 * x * x + 3 * y * y).ravel()]
    return execute(
        "3d.cloud_normals", {"cloud": PointCloud(xyz, frame="model")}, {"neighbors": 12}
    )["cloud"]


def test_native_cloud_normals_icp_and_ppf():
    source = oriented_surface()
    np.testing.assert_allclose(np.linalg.norm(source.normals, axis=1), 1, atol=1e-5)
    shift = np.float32([0.002, 0.003, 0.004])
    target = PointCloud(source.points + shift, frame="scene", normals=source.normals)
    icp = execute("3d.icp", {"source": source, "target": target})
    np.testing.assert_allclose(icp["pose"]["translation"], shift, atol=1e-4)
    assert icp["residual"] < 0.001
    model = execute(
        "3d.ppf_train",
        {"cloud": source},
        {"relative_sampling_step": 0.12, "relative_distance_step": 0.08},
    )["model"]
    matched = execute(
        "3d.ppf_match",
        {"model": model, "scene": target},
        {"scene_sample_step": 0.4, "scene_distance": 0.12},
    )
    assert matched["pose"]["votes"] > 0 and matched["pose"]["target_frame"] == "scene"
    rigid(matched["pose"])


@pytest.mark.parametrize("suffix,binary", [(".ply", False), (".ply", True), (".pcd", False)])
def test_cloud_files_roundtrip_color_normals_units_and_no_overwrite(tmp_path, suffix, binary):
    cloud = PointCloud(
        [[0.02, -0.3, 1], [0.1, 2, 3]],
        "mm",
        "fixture",
        colors=[[255, 0, 1], [0, 255, 19]],
        normals=[[0, 0, 1], [0, 1, 0]],
    )
    path = tmp_path / ("cloud" + suffix)
    write_cloud(path, cloud, binary)
    loaded = read_cloud(path)
    assert loaded.units == "mm" and loaded.frame == "fixture"
    np.testing.assert_array_equal(loaded.colors, cloud.colors)
    np.testing.assert_allclose(loaded.points, cloud.points)
    np.testing.assert_allclose(loaded.normals, cloud.normals)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_cloud(path, cloud, binary)
    assert path.read_bytes() == original
    outputs = execute("input.cloud", parameters={"path": str(path)})
    assert outputs["count"] == 2 and outputs["ticket"]
    destination = tmp_path / ("export" + suffix)
    execute(
        "output.cloud",
        {"cloud": outputs["cloud"]},
        {"path": str(destination), "binary_ply": binary},
    )
    assert read_cloud(destination).frame == "fixture"


def test_cloud_file_rejects_truncation_and_ignores_invalid_xyz(tmp_path):
    path = tmp_path / "bad.ply"
    header = (
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    path.write_text(header + "0 0 1\n", encoding="ascii")
    with pytest.raises(ValueError, match="Truncated"):
        read_cloud(path)
    path.write_text(header + "0 0 1\nnan 0 0\n", encoding="ascii")
    assert len(read_cloud(path).points) == 1


def test_structured_light_identity_shadow_and_pattern_order():
    generated = execute("3d.graycode", parameters={"width": 32, "height": 24})
    p = {"width": 32, "height": 24}
    xy = [[3, 4], [15, 8], [31, 23]]
    out = execute("3d.graycode_decode", {**generated, "points": xy}, p)
    assert out["projector_points"] == xy and all(out["valid"])
    shadow = generated["white"].copy()
    shadow[4, 3] = 0
    out = execute("3d.graycode_decode", {**generated, "white": shadow, "points": xy}, p)
    assert out["projector_points"][0] is None and out["valid"] == [False, True, True]
    group = generated["patterns"]
    execute(
        "3d.graycode_decode",
        {"patterns": FlowGroup(group.items[::-1], group.labels)},
        p,
        state="error",
    )


def test_aruco_pose_recovers_marker_id_and_depth():
    image = np.full((180, 320), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image[42:138, 112:208] = cv2.aruco.generateImageMarker(dictionary, 7, 96)
    out = execute("3d.aruco_pose", {"image": image, "camera": pinhole(320, 180, 200.0)})
    assert out["pose"]["marker_id"] == 7
    assert out["pose"]["translation"][2] == pytest.approx(0.04 * 200 / 96, rel=0.05)
    assert out["overlay"].shape == (180, 320, 3)


def depth_surface():
    yy, xx = np.mgrid[:72, :96]
    return DepthArray(1 + 0.1 * np.sin(xx * 0.13) * np.cos(yy * 0.18))


@pytest.mark.parametrize("mode", ["depth", "rgb", "rgbd"])
def test_ptcloud_odometry_identical_frame_transform(mode):
    depth = depth_surface()
    image = np.random.default_rng(2).integers(0, 256, (72, 96, 3), np.uint8)
    out = execute(
        "3d.odometry",
        {
            "source_depth": depth,
            "target_depth": depth,
            "camera": pinhole(),
            "source_image": image,
            "target_image": image,
        },
        {"mode": mode, "iteration_counts": [3, 5, 7, 10]},
    )
    np.testing.assert_allclose(rigid(out["pose"]), np.eye(4), atol=1e-5)


@pytest.mark.parametrize("method", ["fals", "sri", "cross_product"])
def test_ptcloud_organized_normals_from_metric_plane(method):
    d = DepthArray(np.ones((72, 96), np.float32), "m")
    out = execute("3d.depth_normals", {"depth": d, "camera": pinhole()}, {"method": method})
    cloud = out["cloud"]
    assert len(cloud.points) > 1000
    assert np.median(np.abs(cloud.normals[:, 2])) > 0.99
    assert cloud.source_size == (96, 72)


def test_tsdf_batch_fuses_posed_metric_depth_and_restores_opencl_state():
    d = DepthArray(np.ones((72, 96), np.float32), "m")
    group = FlowGroup((ValuePacket(d, {}), ValuePacket(d.copy(), {})), ("a", "b"))
    previous = cv2.ocl.useOpenCL()
    out = execute(
        "3d.tsdf_fuse",
        {
            "frames": group,
            "camera": pinhole(),
            "poses": [
                pose_record(np.eye(4), "camera", "world"),
                pose_record(translation(0.02), "camera", "world"),
            ],
        },
    )
    assert cv2.ocl.useOpenCL() == previous
    assert out["cloud"].frame == "world" and len(out["cloud"].points) > 100
    assert np.median(out["cloud"].points[:, 2]) == pytest.approx(1.0, abs=0.025)
    assert np.nanmedian(out["depth"]) == pytest.approx(1.0, abs=0.03)


def test_cloud_toolblock_and_headless_summary(tmp_path):
    r = builtin_registry()
    g = rgbd_cloud_graph(r)
    tool = encapsulate(g, ["filter", "sample", "normals"], "Cloud preparation", r)
    assert any(p.kind == "cloud" for p in tool.inputs)
    path = tmp_path / "cloud_tool.json"
    g.save(path)
    report = tmp_path / "report.json"
    assert run_headless([str(path), "--profile", "lean", "--report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["passed"] and data["runs"][0]["blocks"]
    cloud = next(b for b in data["runs"][0]["blocks"] if b["id"] == "project")["outputs"]["cloud"]
    assert cloud["model"] == "point_cloud_v1" and cloud["count"] > 0 and "points" not in cloud


def test_offscreen_cloud_viewer_renders_retained_result(app, load_example):
    window = MainWindow()
    load_example(window, "rgbd_cloud")
    window.graph.run()
    window.show_cloud_viewer()
    viewer = window.cloud_viewer
    viewer.refresh()
    assert viewer.choice.count() >= 3 and viewer.cloud is not None
    assert "points" in viewer.info.text() and not viewer.image.pixmap().isNull()
    previous = viewer.image.pixmap().cacheKey()
    viewer.yaw.setValue(80)
    assert viewer.image.pixmap().cacheKey() != previous
    viewer.close()
    window.dirty = False
    window.close()


@pytest.mark.parametrize("algorithm", ["bm", "sgbm"])
def test_stereo_confidence_rejects_occluded_pixels_and_keeps_the_left_frame(algorithm):
    pair = stereo_pair()
    rectified = execute("3d.stereo_rectify", {k: pair[k] for k in ("left", "right", "rig")})
    left, right = rectified["left"], rectified["right"].copy()
    right[:, 60:90] = 0
    params = {"algorithm": algorithm, "num_disparities": 32, "wls": True, "wls_depth_radius": 3}
    block = builtin_registry().create("3d.disparity", parameters=params)
    inputs = {"left": left, "right": right}
    baseline = block.execute(inputs)
    assert baseline is not None and baseline.passed, block.error
    block.parameters["minimum_wls_confidence"] = 0.8
    filtered = block.execute(inputs)
    assert filtered is not None and filtered.passed, block.error
    confidence = filtered.outputs["confidence"]
    assert confidence.shape == left.shape[:2] and confidence.dtype == np.float32
    assert np.isfinite(confidence).all() and confidence.min() >= 0 and confidence.max() <= 1
    np.testing.assert_array_equal(confidence, baseline.outputs["confidence"])
    kept = filtered.outputs["valid_mask"] != 0
    initial = baseline.outputs["valid_mask"] != 0
    np.testing.assert_array_equal(kept, initial & (confidence >= 0.8))
    assert kept.sum() < initial.sum()
    assert np.isnan(filtered.outputs["disparity"][~kept]).all()
    assert np.nanmedian(filtered.outputs["disparity"]) == pytest.approx(8, abs=0.1)
    assert filtered.payloads["confidence"]["spatial"] == filtered.payloads["disparity"]["spatial"]
