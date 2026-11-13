"""Native parity, independent motion truth, sequence boundaries and honest loss events."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.color_tracking import CRITERIA, ColorDistributionTracker
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.recipes import RecipeCatalog, reference
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/color_region_tracking"


def make(**parameters):
    return ColorDistributionTracker(parameters={"initial_bounds": [30, 30, 24, 16], **parameters})


def target(dx=0, dy=0):
    image = np.full((96, 144, 3), 35, np.uint8)
    image[30 + dy : 46 + dy, 30 + dx : 54 + dx] = (27, 200, 27)
    return image


def run(block, image=None, **inputs):
    result = block.execute({"image": target() if image is None else image, **inputs})
    assert result is not None, block.error
    return result.outputs


def state(block):
    return next(iter(block.memory["streams"].values()))


@pytest.mark.parametrize("method", ["mean_shift", "cam_shift"])
@pytest.mark.parametrize("channels", ["hue", "hue_saturation"])
@pytest.mark.parametrize("termination", CRITERIA)
def test_native_histogram_backprojection_window_and_termination(method, channels, termination):
    block = make(method=method, histogram_channels=channels, termination=termination)
    assert not run(block)["ready"] and block.state == "waiting"
    memory = block.snapshot_runtime()
    saved = next(iter(memory["streams"].values()))
    image = target(2, 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    used, bins, ranges = (
        ([0], [32], [0, 180]) if channels == "hue" else ([0, 1], [32, 32], [0, 180, 0, 256])
    )
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[30:46, 30:54] = 255
    taught = cv2.calcHist([cv2.cvtColor(target(), cv2.COLOR_BGR2HSV)], used, mask, bins, ranges)
    taught *= 255 / taught.max()
    np.testing.assert_array_equal(saved["histogram"], taught)
    expected = cv2.calcBackProject([hsv], used, taught, ranges, 1)
    expected[hsv[..., 1] < 40] = 0
    criteria = (CRITERIA[termination], 20, 1.0)
    if method == "mean_shift":
        iterations, bounds = cv2.meanShift(expected, saved["bounds"], criteria)
    else:
        ellipse, bounds = cv2.CamShift(expected, saved["bounds"], criteria)
    out = run(block, image)
    assert out["ready"] and block.state == "pass"
    assert out["bounds"] == list(bounds)
    np.testing.assert_array_equal(out["probability"], expected)
    if method == "mean_shift":
        assert out["status"]["native_iterations"] == iterations
    else:
        np.testing.assert_allclose(out["geometry"]["center"], ellipse[0])
        np.testing.assert_allclose(out["geometry"]["axes"], ellipse[1])
    np.testing.assert_array_equal(state(block)["histogram"], taught)
    assert out["histogram"] is NO_VALUE  # expensive serialization is opt-in


def test_loss_withholds_primary_events_and_expiry_requires_rising_reseed():
    block = make(max_misses=2, learning_rate=0.25)
    run(block, reseed=True)
    original = state(block)["track_id"]
    assert run(block, reseed=True)["ready"]
    assert state(block)["track_id"] == original
    snapshot = block.snapshot_runtime()
    for _ in range(2):
        out = run(block, np.zeros_like(target()))
        assert not out["ready"] and out["image"] is NO_VALUE and out["pose"] is NO_VALUE
        assert block.state == "fail" and not block.last_result.passed
        assert not block.current_emissions.get("image") and not block.current_emissions.get("pose")
        assert block.current_emissions["status"] and block.last_result.fail_reasons
        assert state(block)["bounds"] == next(iter(snapshot["streams"].values()))["bounds"]
    assert run(block)["status"]["phase"] == "expired"
    assert run(block, reseed=True)["status"]["phase"] == "initialized"
    new_id = state(block)["track_id"]
    assert new_id != original
    assert run(block, reseed=True)["ready"] and state(block)["track_id"] == new_id


def test_temporary_miss_keeps_window_and_model_then_recovers():
    block = make(method="mean_shift", learning_rate=0.5)
    run(block)
    before = block.snapshot_runtime()
    out = run(block, np.zeros_like(target()))
    assert out["status"]["phase"] == "lost"
    old = next(iter(before["streams"].values()))
    np.testing.assert_array_equal(state(block)["histogram"], old["histogram"])
    assert state(block)["bounds"] == old["bounds"] and state(block)["model_updates"] == 0
    out = run(block, target(1, 1))
    assert out["ready"] and out["status"]["misses"] == 0
    assert out["geometry"]["track_id"] == old["track_id"]
    assert out["status"]["model_updates"] == 1


@pytest.mark.parametrize(
    "gate,reason",
    [
        ({"minimum_confidence": 1.0}, "low_probability"),
        ({"minimum_support_fraction": 1.0}, "insufficient_color_support"),
        ({"maximum_area_ratio": 0.3}, "scale_limit"),
        ({"maximum_center_jump": 0.1}, "motion_jump"),
    ],
)
def test_rejected_updates_do_not_commit_candidate(gate, reason):
    block = make(**gate)
    run(block)
    original = block.snapshot_runtime()
    out = run(block, target(2, 1))
    assert not out["ready"] and out["status"]["reason"] == reason
    old = next(iter(original["streams"].values()))
    assert state(block)["bounds"] == old["bounds"] and state(block)["center"] == old["center"]
    np.testing.assert_array_equal(state(block)["histogram"], old["histogram"])


@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba", "hsv"])
def test_color_encoding_alpha_and_current_ticket_preserve_original_pixels(encoding):
    native = target()
    if encoding in ("bgra", "rgba"):
        native = np.dstack([native, np.full(native.shape[:2], 200, np.uint8)])
    if encoding in ("rgb", "rgba"):
        native = native[..., [2, 1, 0] + ([3] if encoding == "rgba" else [])]
    elif encoding == "hsv":
        native = cv2.cvtColor(native, cv2.COLOR_BGR2HSV)
    clock = TicketClock()
    context = {"spatial": frame([144, 96], "camera"), "lot": "A"}
    block = make()
    for index in range(2):
        ticket = clock.issue("camera")
        context["ticket"] = ticket
        image = ImageArray(native.copy(), encoding, context)
        out = run(block, image)
        assert out["status"]["current_ticket"] == ticket
    np.testing.assert_array_equal(out["image"], native)
    assert image_format(out["image"]) == encoding and out["image"].dtype == np.uint8
    assert block.last_result.payloads["pose"]["ticket"] == ticket
    assert block.last_result.payloads["image"]["lot"] == "A"
    assert out["pose"]["target_payload"]["spatial"] == context["spatial"]


def test_editable_polygon_seed_intersects_mask_region_and_hsv_alpha_filters():
    image = ImageArray(np.dstack([target(), np.full((96, 144), 255, np.uint8)]), "bgra")
    image[30:33, :, 3] = 0
    image[33:36, :, :3] = 255  # low saturation
    image[36:39, :, :3] = 0  # low value
    mask = np.zeros((96, 144), np.uint8)
    mask[30:46, 30:48] = 255
    seed = {
        "model": "image_region_v1",
        "image_size": [144, 96],
        "shape": "polygon",
        "vertices": [[30, 30], [53, 30], [53, 45], [30, 45]],
    }
    region = {
        "model": "image_region_v1",
        "image_size": [144, 96],
        "shape": "rectangle",
        "rectangle": [30, 30, 24, 14],
    }
    block = make(initial_bounds=[], minimum_area_ratio=0.1)
    out = run(block, image, mask=mask, region=region, seed=seed)
    assert out["status"]["seed_pixels"] == 18 * 5
    out = run(block, image, mask=mask, region=region)
    assert not out["probability"][:39].any() and not out["probability"][44:].any()
    assert not out["probability"][:, 48:].any()


def test_isotropic_geometry_withholds_principal_axis_pose():
    image = np.zeros((96, 144, 3), np.uint8)
    cv2.circle(image, (45, 42), 10, (27, 200, 27), -1)
    block = make(initial_bounds=[30, 30, 30, 24], pose_mode="principal_axis")
    run(block, image)
    out = run(block, image)
    assert out["ready"] and not out["geometry"]["orientation_valid"]
    assert out["pose"] is NO_VALUE and not block.current_emissions.get("pose")
    assert out["tracked_region"]["shape"] == "ellipse"


def test_seed_required_and_insufficient_seed_are_distinct_from_warmup():
    block = make(initial_bounds=[])
    assert run(block)["status"]["phase"] == "seed_required"
    assert block.state == "waiting"
    circle = {
        "model": "image_region_v1",
        "image_size": [144, 96],
        "shape": "circle",
        "center": [42, 38],
        "radius": 5,
    }
    out = run(block, np.zeros_like(target()), seed=circle)
    assert out["status"]["phase"] == "seed_rejected" and block.state == "fail"
    assert "histogram" not in state(block)
    out = run(block, seed=circle)
    assert out["status"]["phase"] == "initialized"
    assert run(block)["ready"]


def test_recipe_and_grid_change_reinitialize_instead_of_reusing_model():
    block = make()
    catalog = RecipeCatalog()
    first, second = catalog.put("A", {}), catalog.put("B", {})
    for recipe in (first, first, second):
        image = ImageArray(target(), "bgr", {"recipe_ref": reference(recipe)})
        out = run(block, image)
    assert out["status"]["phase"] == "initialized" and out["activations"] == 1
    image.metadata["spatial"] = frame([144, 96], "new-camera")
    assert run(block, image)["status"]["phase"] == "initialized"


def test_copyable_streams_context_reset_gap_and_opt_in_histogram():
    block = make(max_gap_ms=100)
    block.now = lambda: 1.0
    run(block, stream="left")
    run(block, stream="right")
    assert len(block.memory["streams"]) == 2
    clone = make(max_gap_ms=100)
    clone.copy_runtime_from(block)
    clone.now = lambda: 1.05
    clone.exposed_outputs.append("histogram")
    out = run(clone, stream="left")
    assert out["ready"] and out["histogram"]["normalization"] == "peak_255"
    assert list(block.memory["streams"].values())[0]["activations"] == 1
    assert state(block)["histogram"] is not state(clone)["histogram"]
    clone.now = lambda: 2.0
    assert run(clone, stream="left")["status"]["phase"] == "initialized"
    assert run(clone, reset=True, stream="left")["activations"] == 1
    clone.parameters["hue_bins"] = 16
    assert run(clone, stream="left")["activations"] == 1
    clone.close()
    assert len(block.memory["streams"]) == 2


@pytest.mark.parametrize(
    "case", ["gray", "depth", "hsv", "bounds", "reseed", "frame", "limits", "pose", "budget"]
)
def test_bad_inputs_are_contained(case):
    image, extra, parameters = target(), {}, {}
    if case == "gray":
        image = image[..., 0]
    elif case == "depth":
        image = image.astype(np.uint16)
    elif case == "hsv":
        image = ImageArray(np.full_like(image, 200), "hsv")
    elif case == "bounds":
        extra["seed"] = [140, 30, 24, 16]
    elif case == "reseed":
        extra["reseed"] = "yes"
    elif case == "frame":
        image = ImageArray(image, "bgr", {"spatial": frame([144, 96], "A")})
        extra["seed"] = ValuePacket([30, 30, 24, 16], {"spatial": frame([144, 96], "B")})
    elif case == "limits":
        parameters = {"minimum_value": 200, "maximum_value": 100}
    elif case == "pose":
        parameters = {"method": "mean_shift", "pose_mode": "principal_axis"}
    else:
        parameters = {"max_bytes_mb": 0.001}
    block = make(**parameters)
    assert block.execute({"image": image, **extra}) is None
    assert block.state == "error" and block.error and not block.current_emissions


@pytest.mark.parametrize("passed,expected", [(True, "waiting"), (False, "fail")])
def test_explicit_failed_result_is_not_hidden_by_withheld_outputs(passed, expected):
    class Withheld(Block):
        inputs = ()
        outputs = (Port("image"),)

        def process(self, inputs):
            return Result({"image": NO_VALUE}, passed=passed)

    block = Withheld()
    assert block.execute({}) is not None
    assert block.state == expected


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("name", ["mean_shift", "cam_shift", "loss_and_expiry"])
def test_runnable_examples_against_independent_truth_and_xml(name, profile, tmp_path):
    truth = json.loads((EXAMPLE / "truth.json").read_text())
    graph = Graph.load(EXAMPLE / f"{name}.json", builtin_registry())
    sequence = truth[
        {"mean_shift": "translation", "cam_shift": "scale", "loss_and_expiry": "loss"}[name]
    ]
    areas = []
    try:
        graph.set_run_profile(profile)
        for record in sequence:
            graph.run()
            block = graph.blocks["tracker"]
            assert block.last_result is not None, block.error
            result, out = block.last_result, block.last_result.outputs
            assert bool(result.drawings) == (profile == "debug")
            index = record["index"]
            if not index:
                assert block.state == "waiting" and not out["ready"]
            elif name == "loss_and_expiry" and index >= 8:
                assert block.state == "fail" and not out["ready"]
                assert out["image"] is NO_VALUE and out["geometry"] is NO_VALUE
                assert out["status"]["phase"] == ("expired" if index == 11 else "lost")
            else:
                assert block.state == "pass" and out["ready"]
                assert (
                    np.linalg.norm(np.asarray(out["geometry"]["center"]) - record["center"])
                    <= truth["center_tolerance_pixels"]
                )
                if name != "mean_shift":
                    assert out["pose"]["ambiguous_180"]
                    assert abs(out["pose"]["angle_degrees"] + 25) < 5
                    areas.append(out["geometry"]["area"])
        if areas:
            assert areas[-1] / areas[0] >= truth["camshift_minimum_scale_growth"]
        path = tmp_path / "tracker.sequence.color_tracker.xml"
        save_block_xml(block, path)
        assert read_block_xml(path) == block.to_project_item()
    finally:
        graph.close()
