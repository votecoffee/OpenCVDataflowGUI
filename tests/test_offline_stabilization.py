"""Check finite look-ahead, scene boundaries, and replay contracts."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame

TYPE = "sequence.stabilize_batch"


def _texture(seed=314):
    image = np.random.default_rng(seed).integers(0, 256, (112, 160), np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0.9)


def _shift(image, x):
    return cv2.warpAffine(
        image,
        np.array([[1, 0, x], [0, 1, 0]], dtype=float),
        image.shape[1::-1],
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _group(*images):
    items = tuple(
        ValuePacket(image, {"spatial": frame(image.shape[1::-1], "raw")}) for image in images
    )
    return FlowGroup(items, tuple(map(str, range(len(items)))))


@pytest.mark.local_examples
def test_saved_offline_replay_reduces_jitter_and_preserves_frame_tickets():
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools/offline_camera_stabilization/offline_camera_stabilization.json"
    )
    graph = Graph.load(project, builtin_registry())
    try:
        graph.set_run_profile("lean")
        smooth, replay = graph.blocks["smooth"], graph.blocks["replay"]
        for _ in range(47):
            graph.run()
            assert smooth.state == replay.state == "inactive"
        graph.run()
        assert smooth.state == replay.state == "pass", smooth.error
        assert graph.sources_exhausted()
        out = smooth.last_result.outputs
        report = out["report"]
        assert report["status"] == "complete" and report["segment_starts"] == [0]
        frames = report["frames"]
        assert len(frames) == 48
        assert frames[0]["window"] == [0, 5] and frames[-1]["window"] == [43, 48]
        raw, corrected = (
            np.array([record[name] for record in frames])
            for name in ("observed_pose", "smoothed_pose")
        )
        rms = [
            np.sqrt(np.mean(np.sum(np.diff(path[:, :2], n=2, axis=0) ** 2, axis=1)))
            for path in (raw, corrected)
        ]
        assert rms[1] < 0.25 * rms[0]
        assert min(record["coverage_fraction"] for record in frames) > 0.93
        emitted = replay.current_emissions["value"]
        assert len(emitted) == 48
        for index, (image, mask, transform, packet) in enumerate(
            zip(
                out["corrected"].items,
                out["coverage"].items,
                out["transforms"].items,
                emitted,
                strict=True,
            )
        ):
            assert image.payload["ticket"] == mask.payload["ticket"] == transform.payload["ticket"]
            assert packet.payload["ticket"] == image.payload["ticket"]
            assert packet.payload["video_frame"]["frame_index"] == index
            spatial = image.payload["spatial"]
            assert spatial == mask.payload["spatial"] == transform.value["target_spatial"]
            assert np.all(image.value[mask.value == 0] == 0)
            center = (np.asarray(spatial["image_size"]) - 1) / 2
            raw_center = [*(center + raw[index, :2]), 1]
            to_root = np.asarray(spatial["to_root"]) @ np.asarray(
                transform.value["source_to_target"]
            )
            np.testing.assert_allclose(to_root @ raw_center, [*center, 1], atol=1e-9)
    finally:
        graph.close()


def test_centered_smoothing_uses_future_frames_and_preserves_16_bit_encoding():
    base = _texture().astype(np.uint16) * 257
    images = [
        ImageArray(np.repeat(_shift(base, x)[..., None], 3, axis=2), "rgb") for x in (0, 0, 6, 0, 0)
    ]
    block = builtin_registry().create(
        TYPE, parameters={"motion_model": "translation", "radius_frames": 1, "sigma_frames": 1}
    )
    result = block.execute({"frames": _group(*images)})
    assert result is not None and result.passed, block.error
    record = result.outputs["report"]["frames"][1]
    assert record["window"] == [0, 3] and abs(record["observed_pose"][0]) < 0.01
    expected = 6 * np.exp(-0.5) / (1 + 2 * np.exp(-0.5))
    assert abs(record["smoothed_pose"][0] - expected) < 0.05
    corrected = result.outputs["corrected"].items[1].value
    assert corrected.dtype == np.uint16 and image_format(corrected) == "rgb"
    assert result.outputs["corrected"].labels == ("0", "1", "2", "3", "4")


def test_rejected_motion_splits_the_smoothing_window_and_coordinate_root():
    a, b = _texture(), _texture(315)
    blank = np.zeros_like(a)
    group = _group(a, _shift(a, 4), blank, b, _shift(b, -4))
    block = builtin_registry().create(TYPE, parameters={"motion_model": "translation"})
    result = block.execute({"frames": group})
    assert result is not None and result.passed, block.error
    report = result.outputs["report"]
    assert report["segment_starts"] == [0, 2, 3]
    assert [record["window"] for record in report["frames"]] == [[0, 2]] * 2 + [[2, 3]] + [
        [3, 5]
    ] * 2
    items = result.outputs["corrected"].items
    roots = [item.payload["spatial"]["root_id"] for item in items]
    assert len(set(roots)) == 3 and roots[0] == roots[1] and roots[3] == roots[4]
    np.testing.assert_array_equal(items[2].value, blank)


def test_coverage_failure_and_cancellation_withhold_the_entire_batch():
    image = _texture()
    group = _group(image, _shift(image, 10))
    block = builtin_registry().create(
        TYPE, parameters={"motion_model": "translation", "min_coverage_fraction": 1.0}
    )
    failed = block.execute({"frames": group})
    assert failed is not None and not failed.passed, block.error
    assert failed.outputs["corrected"] is failed.outputs["coverage"] is NO_VALUE
    assert failed.outputs["report"]["status"] == "insufficient_coverage"
    assert failed.outputs["report"]["failed_frame"] == 0
    assert failed.fail_reasons[0]["code"] == "offline_stabilization_insufficient_coverage"
    cancelled = block.execute({"frames": group}, cancelled=lambda: True)
    assert cancelled is not None and not cancelled.passed
    assert cancelled.outputs["corrected"] is NO_VALUE
    assert cancelled.outputs["report"]["status"] == "cancelled"
    assert block.execute({"frames": _group(image, image[:-1])}) is None
    assert block.error_kind == "invalid" and "match in size" in block.error
