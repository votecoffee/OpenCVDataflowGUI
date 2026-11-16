"""Original PNG/FFV1 clips, independent corrections and saved-project replay."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow import offline_stabilization
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/motion_trajectory"
TRUTH = json.loads((HERE / "truth.json").read_bytes())


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_clips_replay_exact_pixels_and_current_acquisitions(case, profile, monkeypatch):
    expected = TRUTH["cases"][case]
    graph = Graph.load(HERE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    if case != "native_fit_capture":
        monkeypatch.setattr(
            offline_stabilization,
            "_estimate_pose",
            lambda *a: pytest.fail("Loaded replay refitted motion"),
        )
    try:
        for _ in range(5):
            graph.run()
        block = graph.blocks["replay"]
        assert block.state == expected["expected_state"], block.error
        result = block.last_result
        if expected["expected_state"] == "error":
            assert result is None and block.error_kind == "invalid"
            assert not any(block.current_emissions.values())
            return
        assert result is not None
        if case == "invalid_reject":
            assert result.outputs["corrected"] is result.outputs["image"] is NO_VALUE
            assert not block.current_emissions["corrected"]
            return
        group = graph.blocks["frames"].last_result.outputs["group"]
        trajectory = graph.blocks["trajectory"].last_result.outputs["trajectory"]
        indices = [i for i in range(5) if not (expected["record"] == "invalid" and i == 2)]
        assert result.outputs["report"]["emitted_count"] == len(indices)
        base = group.items[0].value
        y, x = np.indices(base.shape[:2])
        for packet, mask, transform, index in zip(
            result.outputs["corrected"].items,
            result.outputs["coverage"].items,
            result.outputs["transforms"].items,
            indices,
            strict=True,
        ):
            if case == "native_fit_capture":
                reference = graph.blocks["fit"].last_result.outputs["corrected"].items[index].value
                expected_mask = (
                    graph.blocks["fit"].last_result.outputs["coverage"].items[index].value
                )
            else:
                dx, dy = TRUTH["shifts"][index]
                valid = (x + dx >= 0) & (x + dx < 160) & (y + dy >= 0) & (y + dy < 112)
                reference = base.copy()
                reference[~valid] = 0
                expected_mask = valid.astype(np.uint8) * 255
            np.testing.assert_array_equal(packet.value, reference)
            np.testing.assert_array_equal(mask.value, expected_mask)
            assert packet.value.dtype == base.dtype
            assert packet.payload["ticket"] == group.items[index].payload["ticket"]
            assert packet.payload["motion_trajectory"]["index"] == index
            assert packet.payload["spatial"]["root_id"].startswith(
                "trajectory:" + trajectory["trajectory_id"] + ":"
            )
            assert transform.value["source_spatial"] == group.items[index].payload["spatial"]
            if expected["video"]:
                assert packet.payload["video_frame"]["frame_index"] == index
                assert packet.payload["motion_trajectory"]["time"]["timestamp_ms"] == pytest.approx(
                    index * 40.0, abs=1e-6
                )
        serial = graph.blocks["serial"]
        assert len(serial.current_emissions["value"]) == len(indices)
        assert [p.payload["ticket"] for p in serial.current_emissions["value"]] == [
            group.items[i].payload["ticket"] for i in indices
        ]
        if expected["record"] == "invalid":
            assert result.outputs["report"]["omitted_indices"] == [2]
            assert result.passed == (expected["invalid_frames"] == "skip")
        # Replay owns no mutable cursor or native fit state across a graph reset.
        block.reset_runtime()
        again = block.execute({"frames": group, "trajectory": trajectory})
        np.testing.assert_array_equal(again.outputs["image"], result.outputs["image"])
    finally:
        graph.close()


def test_regeneration_matches_every_original_image_video_record_and_program(tmp_path):
    spec = importlib.util.spec_from_file_location("trajectory_example", HERE / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    for source in HERE.glob("*.json"):
        assert (tmp_path / source.name).read_bytes() == source.read_bytes(), source
    for name, sha in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == sha
