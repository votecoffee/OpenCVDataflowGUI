"""Check the sensor model, motion rejection, support and saved burst quality."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import FlowGroup, NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.translation_reconstruction import BoxObservation

TYPE = "sequence.multi_frame_super_resolution"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/multi_frame_super_resolution"
SHIFTS = [[0.5, 0.0], [0.0, 0.5], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.0]]


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.fixture(scope="module")
def frames():
    return [cv2.imread(str(EXAMPLE / f"assets/frame_{i:02}.png"), 0) for i in range(1, 6)]


def group_of(images):
    clock = TicketClock("super-resolution")
    packets = []
    for i, image in enumerate(images):
        packet = {
            "spatial": frame(list(image.shape[1::-1]), "camera"),
            "ticket": clock.issue("camera", "fixed", f"sample {i}"),
        }
        packets.append(ValuePacket(image, packet))
    return FlowGroup(tuple(packets), ("camera",) * len(packets))


@pytest.mark.parametrize("scale", (2, 4))
def test_aperture_forward_and_adjoint_agree_and_exclude_incomplete_borders(scale):
    rng = np.random.default_rng(811)
    low = rng.normal(size=(8, 11, 3)).astype(np.float32)
    high = rng.normal(size=(8 * scale, 11 * scale, 3)).astype(np.float32)
    model = BoxObservation.create(low, scale, (0.37, -0.22))
    assert not model.valid[:, 0].any() and not model.valid[-1].any()
    assert np.vdot(model.forward(high), low) == pytest.approx(np.vdot(high, model.adjoint(low)), abs=2e-5)
    shifted = BoxObservation.create(low, scale, (1 / scale, 0))
    donor = np.concatenate([high[:, :1], high[:, :-1]], axis=1)
    expected = cv2.resize(donor, (11, 8), interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(shifted.forward(high)[shifted.valid], expected[shifted.valid], atol=1e-6)


@pytest.mark.parametrize(("mode", "profile"), [("estimate", "lean"), ("provided", "debug")])
def test_saved_burst_reconstructs_only_after_five_frames(registry, mode, profile):
    graph = Graph.load(EXAMPLE / f"super_resolution_{mode}.json", registry)
    graph.set_run_profile(profile)
    graph.set_branch_workers(3)
    try:
        recon = graph.blocks["reconstruct"]
        for _ in range(4):
            graph.run()
            assert not recon.current_emissions
        graph.run()
        assert graph.sources_exhausted()
        assert recon.state == "pass", recon.error
        result = recon.last_result
        assert result.outputs["image"].shape == (144, 192)
        assert result.payloads["image"]["ticket"]["value"] == "frame_03.png"
        assert result.payloads["image"]["spatial"]["to_root"] == [[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]]
        report = result.outputs["report"]
        assert report["anchor_index"] == 2 and report["window_indices"] == list(range(5))
        assert len(report["phase_bins"]) == 4 and report["accepted_frames"] == 5
        np.testing.assert_allclose([r["shift_pixels"] for r in report["frames"]], SHIFTS, atol=0.01)
        assert graph.blocks["image_rmse"].last_result.outputs["value"] < 3.3
        assert graph.blocks["baseline_rmse"].last_result.outputs["value"] == pytest.approx(20.74, abs=0.04)
        image = result.outputs["image"]
        assert image[51:95, 62].mean() - image[51:95, 65].mean() > 130
    finally:
        graph.close()


def test_anchor_window_selection_and_color_depth_are_preserved(registry, frames):
    planes = []
    for image in frames:
        array = np.dstack((image, image // 2, 255 - image, np.full_like(image, 243)))
        planes.append(ImageArray(array.astype(np.uint16) * 257, "rgba"))
    group = group_of([planes[0], *planes, planes[-1]])
    selection = np.zeros(frames[0].shape, np.uint8)
    selection[20:45, 20:53] = 255
    block = registry.create(TYPE, parameters={
        "target_index": 3, "alignment": "provided", "shifts": [[40, 40], *SHIFTS, [40, 40]],
        "iterations": 8,
    })
    payload = group.items[3].payload
    result = block.execute({"frames": group, "mask": ValuePacket(selection, payload)})
    assert result and result.passed, block.error
    outputs = result.outputs
    assert outputs["report"]["window_indices"] == [1, 2, 3, 4, 5]
    assert result.payloads["image"]["ticket"] == payload["ticket"]
    image, baseline = outputs["image"], outputs["baseline"]
    assert image.dtype == np.uint16 and image.color_format == "rgba"
    np.testing.assert_array_equal(image[..., 3], baseline[..., 3])
    expected = cv2.resize(selection, (192, 144), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(outputs["processed_mask"], expected)
    np.testing.assert_array_equal(image[expected == 0], baseline[expected == 0])
    assert np.any(image[expected != 0] != baseline[expected != 0])


def test_rolling_batch_reconstructs_each_center_after_warmup(registry, frames):
    burst = group_of([*frames, *frames[:2]])
    batch = registry.create("flow.batch", parameters={"size": 5, "stride": 1})
    block = registry.create(TYPE, parameters={"iterations": 4, "minimum_phase_bins": 4})
    centers = []
    group_tickets = []
    for index, packet in enumerate(burst.items):
        queued = batch.execute({"item_1": packet})
        if index < 4:
            assert queued.outputs["group"] is NO_VALUE
            continue
        assert queued.outputs["queued"] == 4
        group_tickets.append(queued.payloads["group"]["ticket"]["sequence"])
        group = ValuePacket(queued.outputs["group"], queued.payloads["group"])
        result = block.execute({"frames": group})
        assert result and result.passed, block.error
        centers.append(result.payloads["image"]["ticket"]["value"])
    assert centers == ["sample 2", "sample 3", "sample 4"]
    assert len(set(group_tickets)) == 3


@pytest.mark.parametrize(
    ("shifts", "reason"),
    [
        ([[0.0, 0.0]] * 5, "super_resolution_insufficient_subpixel_phases"),
        ([[5.0, 0.0], [0.0, 5.0], [0.0, 0.0], [5.0, 5.0], [-5.0, 0.0]], "super_resolution_too_few_aligned_frames"),
    ],
)
def test_withholds_reconstruction_when_alignment_or_phase_support_fails(registry, frames, shifts, reason):
    block = registry.create(TYPE, parameters={"alignment": "provided", "shifts": shifts})
    result = block.execute({"frames": group_of(frames)})
    assert result and not result.passed
    assert result.outputs["image"] is NO_VALUE
    assert result.outputs["baseline"].shape == (144, 192)
    assert not result.outputs["processed_mask"].any()
    assert [failure["code"] for failure in result.fail_reasons] == [reason]
    assert result.outputs["report"]["failure"] == reason
    assert result.outputs["coverage"][40, 40] == result.outputs["report"]["accepted_frames"]


def test_huber_residual_limits_ghost_from_a_corrupted_observation(registry, frames):
    corrupted = frames[0].copy()
    corrupted[19:37, 35:49] = 235
    group = group_of([corrupted, *frames[1:]])
    truth = cv2.imread(str(EXAMPLE / "assets/truth.png"), 0)[40:70, 73:96].astype(float)
    errors = []
    for limit in (1.0, 0.06):
        block = registry.create(TYPE, parameters={
            "alignment": "provided", "shifts": SHIFTS, "iterations": 120, "residual_limit": limit,
        })
        result = block.execute({"frames": group})
        assert result and result.passed, block.error
        errors.append(np.sqrt(np.mean((result.outputs["image"][40:70, 73:96] - truth) ** 2)))
    assert errors[1] < 8 and errors[1] < errors[0] / 2
