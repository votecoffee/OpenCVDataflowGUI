"""Runnable registered-view projects checked against independent placement truth."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/registered_view_replay"
TRUTH = json.loads((HERE / "truth.json").read_bytes())


def reference(expected, mask):
    frames, supports = [], []
    for index, filename in zip(expected["order"], expected["filenames"], strict=True):
        image = cv2.imread(str(HERE / "assets" / filename), cv2.IMREAD_UNCHANGED)
        output = np.zeros((128, 288, *image.shape[2:]), image.dtype)
        valid = np.zeros((128, 288), bool)
        output[:, index * 48 : index * 48 + 192] = image
        valid[:, index * 48 : index * 48 + 192] = mask
        output[~valid] = 0
        frames.append(output)
        supports.append(valid)
    union, common = np.logical_or.reduce(supports), np.logical_and.reduce(supports)
    bounds = [0, 0, 288, 128]
    if expected["crop"] != "canvas":
        ys, xs = np.where(common if expected["crop"] == "intersection" else union)
        bounds = list(
            map(int, (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        )
    x, y, w, h = bounds
    common = common[y : y + h, x : x + w]
    for j in range(len(frames)):
        frames[j] = frames[j][y : y + h, x : x + w].copy()
        supports[j] = supports[j][y : y + h, x : x + w].copy()
        if expected["crop"] == "intersection":
            supports[j] &= common
        frames[j][~supports[j]] = 0
    return frames, supports, common, bounds


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_replay_projects_preserve_pixels_and_emit_serial_ticketed_inspections(case, profile):
    graph = Graph.load(HERE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    expected = TRUTH["cases"][case]
    try:
        for _ in range(expected["runs"]):
            graph.run()
        block = graph.blocks["replay"]
        assert block.state == expected["expected_state"], block.error
        result = block.last_result
        if expected["expected_state"] == "error":
            assert result is None and block.error_kind == "invalid"
            assert not any(block.current_emissions.values())
            return
        assert result is not None
        if expected["expected_state"] == "fail":
            assert result.outputs["frames"] is result.outputs["image"] is NO_VALUE
            assert result.outputs["info"]["status"] == "no_common_support"
            assert not block.current_emissions["frames"]
            return
        mask = np.ones((128, 192), bool)
        if expected["masked"]:
            mask = graph.blocks["selection"].last_result.outputs["mask"] != 0
        frames, supports, common, bounds = reference(expected, mask)
        for j, (packet, image, valid) in enumerate(
            zip(result.outputs["frames"].items, frames, supports, strict=True)
        ):
            np.testing.assert_array_equal(packet.value, image)
            np.testing.assert_array_equal(
                result.outputs["coverage"].items[j].value, valid.astype(np.uint8) * 255
            )
            assert packet.payload["ticket"]["value"] == expected["filenames"][j]
            assert (
                packet.payload["registered_replay"]["source_id"]
                == ("left", "middle", "right")[expected["order"][j]]
            )
            spatial = packet.payload["spatial"]
            assert spatial["root_size"] == [288, 128]
            assert spatial["root_id"] == "mosaic:" + result.outputs["info"]["layout_id"]
            np.testing.assert_array_equal(
                spatial["to_root"], [[1, 0, bounds[0]], [0, 1, bounds[1]], [0, 0, 1]]
            )
        np.testing.assert_array_equal(result.outputs["common"], common.astype(np.uint8) * 255)
        serial = graph.blocks["serial"]
        assert [
            p.payload["ticket"]["value"] for p in serial.current_emissions["value"]
        ] == expected["filenames"]
        for packet, expected_frame in zip(serial.current_emissions["value"], frames, strict=True):
            np.testing.assert_array_equal(packet.value, expected_frame)
        if "measure" in graph.blocks:
            measure = graph.blocks["measure"]
            assert measure.state == "pass", measure.error
            emitted = measure.current_emissions["selected_count"]
            assert len(emitted) == len(frames)
            assert all(p.value == np.count_nonzero(common) for p in emitted)
            assert [p.payload["ticket"]["value"] for p in emitted] == expected["filenames"]
        if profile == "lean":
            assert not result.graphics and not result.drawings
    finally:
        graph.close()


def test_regeneration_reproduces_every_project_image_and_sealed_layout(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "registered_replay_example", HERE / "build_example.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    for source in HERE.glob("*.json"):
        assert (tmp_path / source.name).read_bytes() == source.read_bytes(), source
    for name, sha in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == sha
