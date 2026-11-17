"""Native refinement controls, reserved-observation ownership and known-camera checks."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.panorama_fitting import partition
from opencv_dataflow.panorama_geometry import project_points, validate_layout
from opencv_dataflow.payloads import NO_VALUE

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/panorama_refinement"
SOURCE = BASE.parent / "rotating_panorama"
FAILED = {"validation_rejected", "disconnected", "insufficient_validation"}
CASES = [
    "reprojection",
    "ray",
    "fixed_intrinsics",
    "initial_only",
    "refine_prior",
    "validation_rejected",
    "disconnected",
    "wave_corrected",
    "insufficient_validation",
]


def camera_example(name):
    return Graph.load(BASE / f"{name}.json", builtin_registry())


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", CASES)
def test_explicit_fit_and_reserved_checks_match_known_cameras(case, profile):
    truth = json.loads((SOURCE / "truth.json").read_bytes())
    graph = camera_example(case)
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["panorama"]
        assert block.last_result is not None, block.error
        assert not any(b.state == "error" for b in graph.processing_blocks())
        result = block.last_result
        report = result.outputs["report"]
        assert result.passed is (case not in FAILED), report
        fit = report["camera_fit"]
        assert fit["validation_excluded_from_camera_fit"]
        assert fit["validation_basis"] == "reserved_keypoints"
        assert all(row["fit"] + row["validation"] <= 1500 for row in fit["features"])
        assert all(row["fit"] > 0 and row["validation"] > 0 for row in fit["features"])
        if case in FAILED:
            assert result.outputs["image"] is NO_VALUE
            assert not block.current_emissions.get("layout")
            if case == "validation_rejected":
                assert report["status"] == "inconsistent_rotation_model"
            if case == "insufficient_validation":
                assert all(row["status"] == "insufficient_matches" for row in report["pairs"])
                assert fit["training_pairs"][1]["inliers"] > 0
            return
        layout = validate_layout(result.outputs["layout"])
        if case == "fixed_intrinsics":
            assert result.outputs["image"].dtype == np.uint16
            assert result.outputs["image"].max() > 255
        for camera in layout["cameras"]:
            K = np.asarray(camera["K"])
            assert abs(K[0, 0] / 420 - 1) < 0.1
            assert abs(K[1, 1] / 420 - 1) < 0.1
            if case in ("fixed_intrinsics", "initial_only", "wave_corrected"):
                np.testing.assert_allclose(K, [[420, 0, 240], [0, 420, 180], [0, 0, 1]], atol=1e-9)
            label = camera["label"]
            yaw = np.deg2rad(truth["yaw_degrees"][label])
            reference_R = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
            delta = np.asarray(camera["camera_to_anchor"]).T @ reference_R
            angle = np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
            assert angle < 3
            # Compare known rays in the pre-wave frame; wave changes common orientation only.
            source_camera = {**camera, "camera_to_panorama": camera["camera_to_anchor"]}
            landmarks = truth["landmarks"][label]
            mapped, valid = project_points(landmarks["pixels"], source_camera, "spherical", 280)
            x, y, z = np.asarray(landmarks["world_rays"]).T
            expected = np.c_[
                280 * np.arctan2(x, z), 280 * (np.pi / 2 + np.arctan2(y, np.hypot(x, z)))
            ]
            assert valid.all()
            assert np.sqrt(np.mean(np.sum((mapped - expected) ** 2, axis=1))) < 5
        assert all(
            row["camera_rms_pixels"] < 3 for row in report["pairs"] if row["status"] == "validated"
        )
        assert graph.blocks["filmstrip"].last_result is not None
        assert bool(result.drawings) is (profile == "debug")
    finally:
        graph.close()


def test_partition_keeps_duplicate_observations_together_and_is_order_independent():
    keys = [
        cv2.KeyPoint(float(x) + 0.2, float(y) + 0.2, size)
        for x in range(12)
        for y in range(9)
        for size in (3, 6)
    ]
    selected = partition(keys, 0.35, 104)
    assert np.array_equal(selected[::2], selected[1::2])
    assert selected.any() and not selected.all()
    np.testing.assert_array_equal(partition(keys[::-1], 0.35, 104), selected[::-1])
    assert not np.array_equal(selected, partition(keys, 0.35, 105))


def test_native_matching_estimation_and_adjustment_never_receive_reserved_centers(monkeypatch):
    from opencv_dataflow import panorama_fitting

    observed = []
    for name in ("BestOf2NearestMatcher", "HomographyBasedEstimator", "BundleAdjusterReproj"):
        original = getattr(cv2.detail, name)

        def constructor(*args, _factory=original, _name=name, **kwargs):
            instance = _factory(*args, **kwargs)

            class Spy:
                def __getattr__(self, key):
                    return getattr(instance, key)

                def check(self, features):
                    sets = []
                    for f in features:
                        keys = list(f.keypoints)
                        assert not partition(keys, 0.35, 104).any()
                        sets.append({tuple(k.pt) for k in keys})
                    observed.append((_name, sets))

                def apply2(self, features):
                    self.check(features)
                    return instance.apply2(features)

                def apply(self, features, *args):
                    self.check(features)
                    return instance.apply(features, *args)

            return Spy()

        monkeypatch.setattr(panorama_fitting.cv2.detail, name, constructor)
    graph = camera_example("reprojection")
    try:
        graph.run()
        assert graph.blocks["panorama"].state == "pass", graph.blocks["panorama"].error
        assert {entry[0] for entry in observed} == {
            "BestOf2NearestMatcher",
            "HomographyBasedEstimator",
            "BundleAdjusterReproj",
        }
        assert observed[0][1] == observed[1][1] == observed[2][1]
    finally:
        graph.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"camera_solver": "automatic", "initial_intrinsics": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
        {"bundle_adjustment": "ray", "refine_focal": False},
        {"bundle_adjustment": "ray", "refine_aspect": True},
        {"bundle_adjustment": "ray", "refine_principal_point": True},
        {"holdout_fraction": 0},
        {"holdout_seed": True},
        {"refine_focal": 1},
        {"initial_intrinsics": [[[1, 2, 3], [0, 1, 2], [0, 0, 1]]]},
        {"camera_features": 32, "minimum_inliers": 33},
    ],
)
def test_bad_fitting_controls_are_rejected_before_execution(settings):
    block = builtin_registry().create(
        "cv.stitch_rotating_panorama", parameters={"camera_solver": "explicit", **settings}
    )
    with pytest.raises(ValueError):
        block.validate_parameters()


@pytest.mark.parametrize("kind", ["count", "outside", "ray_off_center"])
def test_image_dependent_priors_reject_before_detection(kind, monkeypatch):
    K = [[420, 0, 240], [0, 420, 180], [0, 0, 1]]
    graph = camera_example("fixed_intrinsics")
    block = graph.blocks["panorama"]
    if kind == "count":
        block.parameters["initial_intrinsics"] = [K]
    elif kind == "outside":
        K[0][2] = 900
        block.parameters["initial_intrinsics"] = [K] * 3
    else:
        K[0][2] = 241
        block.parameters.update(
            initial_intrinsics=[K] * 3, bundle_adjustment="ray", refine_focal=True
        )
    monkeypatch.setattr(
        cv2.detail,
        "computeImageFeatures2",
        lambda *a: (_ for _ in ()).throw(AssertionError("native detector called")),
    )
    try:
        graph.run()
        assert block.state == "error"
        assert "native detector called" not in block.error
        assert not block.current_emissions.get("image")
    finally:
        graph.close()


def test_explicit_validation_uses_its_partition_cap_instead_of_automatic_feature_limit():
    block = builtin_registry().create(
        "cv.stitch_rotating_panorama",
        parameters={
            "camera_solver": "explicit",
            "verification_features": 32,
            "minimum_inliers": 33,
        },
    )
    block.validate_parameters()
    assert block.last_result is None


def test_original_pixel_priors_survive_work_grid_reduction():
    graph = camera_example("fixed_intrinsics")
    block = graph.blocks["panorama"]
    block.parameters["registration_megapixels"] = 0.1
    try:
        graph.run()
        assert block.state == "pass", block.error
        report = block.last_result.outputs["report"]
        assert report["camera_fit"]["initial_intrinsics_work"][0][0][0] < 420
        for camera in block.last_result.outputs["layout"]["cameras"]:
            np.testing.assert_allclose(
                camera["K"], [[420, 0, 240], [0, 420, 180], [0, 0, 1]], atol=1e-9
            )
    finally:
        graph.close()


def test_native_adjustment_exception_is_contained_and_next_run_recovers(monkeypatch):
    original = cv2.detail.BundleAdjusterReproj
    graph = camera_example("reprojection")
    try:
        monkeypatch.setattr(
            cv2.detail,
            "BundleAdjusterReproj",
            lambda: (_ for _ in ()).throw(cv2.error("adjustment test")),
        )
        graph.run()
        block = graph.blocks["panorama"]
        assert block.state == "error" and "adjustment test" in block.error
        assert not block.current_emissions.get("image")
        monkeypatch.setattr(cv2.detail, "BundleAdjusterReproj", original)
        graph.run()
        assert block.state == "pass" and block.current_emissions["image"]
    finally:
        graph.close()


def test_native_false_result_is_a_failed_inspection_without_numerical_outputs(monkeypatch):
    original = cv2.detail.BundleAdjusterReproj

    class FailedAdjustment:
        def __init__(self):
            self.instance = original()

        def __getattr__(self, name):
            return getattr(self.instance, name)

        def apply(self, *args):
            return False, ()

    monkeypatch.setattr(cv2.detail, "BundleAdjusterReproj", FailedAdjustment)
    graph = camera_example("reprojection")
    try:
        graph.run()
        block = graph.blocks["panorama"]
        assert block.state == "fail" and not block.error
        assert block.last_result.outputs["report"]["status"] == "camera_adjustment_failed"
        assert not block.current_emissions.get("image")
    finally:
        graph.close()


def test_fitting_descriptor_budget_precedes_native_matching(monkeypatch):
    graph = camera_example("reprojection")
    block = graph.blocks["panorama"]
    block.parameters["maximum_verification_pairs"] = 1
    monkeypatch.setattr(
        cv2.detail,
        "BestOf2NearestMatcher",
        lambda *a: (_ for _ in ()).throw(AssertionError("native matcher called")),
    )
    try:
        graph.run()
        assert block.state == "error" and "descriptor budget" in block.error
        assert "native matcher called" not in block.error
        assert not block.current_emissions.get("image")
    finally:
        graph.close()


def test_explicit_controls_and_priors_persist_without_starting_fitting(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QCheckBox, QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml

    graph = camera_example("fixed_intrinsics")
    snapshot = graph.to_dict()
    xml = tmp_path / "cameras.xml"
    export_project_xml(snapshot, xml)
    restored = Graph.from_dict(read_project_xml(xml), builtin_registry())
    assert restored.to_dict() == snapshot
    restored.close()
    monkeypatch.setattr(
        cv2.detail,
        "computeImageFeatures2",
        lambda *a: (_ for _ in ()).throw(AssertionError("unexpected execution")),
    )
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(graph.blocks["panorama"])
        assert isinstance(window.inspector.fields["camera_solver"], QComboBox)
        assert isinstance(window.inspector.fields["bundle_adjustment"], QComboBox)
        assert isinstance(window.inspector.fields["refine_focal"], QCheckBox)
        assert graph.blocks["panorama"].last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()


def test_refinement_examples_regenerate_without_replacing_shared_inputs(tmp_path):
    spec = importlib.util.spec_from_file_location("refinement_examples", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = {case: (BASE / f"{case}.json").read_bytes() for case in CASES}
    module.HERE = tmp_path
    module.SOURCES = SOURCE
    module.build()
    # Save As rebases ordinary source paths; compare all other project data.
    for case in CASES:
        expected = json.loads(before[case])
        actual = json.loads((tmp_path / f"{case}.json").read_bytes())
        for original, rebuilt in zip(expected["blocks"], actual["blocks"], strict=True):
            if original["type"] == "input.file":
                assert "\\" not in original["parameters"]["path"]
                assert (tmp_path / rebuilt["parameters"]["path"]).resolve() == (
                    BASE / original["parameters"]["path"]
                ).resolve()
                rebuilt["parameters"]["path"] = original["parameters"]["path"]
        assert actual == expected
    assert before == {case: (BASE / f"{case}.json").read_bytes() for case in CASES}
