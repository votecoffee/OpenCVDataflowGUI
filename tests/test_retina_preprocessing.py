"""Verify native channels, reproducible resets, and exposure/motion response."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.native_runtime import cpu_execution
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/retina_preprocessing"
TYPE = "sequence.retina"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def test_color_controls_native_parity_and_selected_payload(registry):
    height, width = 49, 65
    spatial = frame([width, height], "camera")
    ticket = TicketClock("retina").issue("camera", "fixed", "first")
    image = np.random.default_rng(892).integers(15, 230, (height, width, 3), np.uint8)
    rgba = np.concatenate((image[..., ::-1], np.full((height, width, 1), 255, np.uint8)), axis=2)
    high_depth = ImageArray(rgba.astype(np.uint16) * 257, "rgba")
    mask = np.zeros((height, width), np.uint8)
    mask[8:42, 10:51] = 255
    selected = mask != 0
    tool = registry.create(
        TYPE,
        parameters={
            "warmup": 1,
            "color_sampling": "diagonal",
            "photoreceptor_time": 1.1,
            "horizontal_gain": 0.07,
            "amacrine_time": 1.5,
            "motion_sensitivity": 0.82,
            "saturate_colors": True,
            "color_saturation": 3.2,
            "normalize_motion": True,
        },
    )
    tool.exposed_outputs += ["detail_raw", "motion_raw"]
    with cpu_execution():
        native = cv2.bioinspired.Retina_create(
            (width, height), True, cv2.bioinspired.RETINA_COLOR_DIAGONAL
        )
    native.setupOPLandIPLParvoChannel(True, True, 0.75, 1.1, 0.53, 0.07, 0.5, 7, 0.75)
    native.setupIPLMagnoChannel(True, 0, 0, 7, 1.5, 0.82, 0, 7)
    native.setColorSaturation(True, 3.2)
    native.run((rgba.astype(np.uint16) * 257)[..., 2::-1].astype(np.float32) * (255 / 65535))
    result = tool.execute(
        {
            "image": ValuePacket(
                high_depth,
                {"spatial": spatial, "ticket": ticket, "alpha_representation": "straight"},
            ),
            "mask": ValuePacket(mask, {"spatial": spatial}),
        }
    )
    assert result and result.passed, tool.error
    detail, motion = (result.outputs[name] for name in ("detail", "motion"))
    assert image_format(detail) == "rgb" and image_format(motion) == "gray"
    expected_detail = native.getParvoRAW().reshape(3, height, width).transpose(1, 2, 0)
    expected_motion = native.getMagnoRAW().reshape(height, width)
    np.testing.assert_array_equal(
        detail[selected], np.rint(np.clip(expected_detail[selected], 0, 255))
    )
    np.testing.assert_array_equal(motion[selected], native.getMagno()[selected])
    np.testing.assert_allclose(result.outputs["detail_raw"][selected], expected_detail[selected])
    np.testing.assert_allclose(result.outputs["motion_raw"][selected], expected_motion[selected])
    for name in ("detail", "motion", "detail_raw", "motion_raw"):
        assert not result.outputs[name][~selected].any()
        assert result.payloads[name]["ticket"] == ticket
        assert result.payloads[name]["spatial"] == spatial
        assert "alpha_representation" not in result.payloads[name]
    assert result.outputs["status"]["motion_mean"] == pytest.approx(
        expected_motion[selected].mean()
    )


@pytest.mark.parametrize("channel", ("detail", "motion"))
def test_channel_selection_and_finite_black_reset(registry, channel):
    black = np.zeros((32, 48), np.uint8)
    tool = registry.create(TYPE, parameters={"warmup": 2, "channels": channel})
    tool.exposed_outputs += ["detail_raw", "motion_raw"]

    def execute(image, index, **options):
        result = tool.execute(
            {
                "image": ValuePacket(image, {"video_frame": {"frame_index": index}}),
                **options,
            }
        )
        assert result and result.passed, tool.error
        return result

    ready = execute(black, 0)
    assert ready.outputs[channel] is NO_VALUE
    ready = execute(black, 1)
    raw = ready.outputs[f"{channel}_raw"].copy()
    assert np.isfinite(raw).all()
    other = "detail" if channel == "motion" else "motion"
    assert ready.outputs[other] is NO_VALUE
    bright = execute(np.full_like(black, 100), 2)
    assert np.isfinite(bright.outputs[f"{channel}_raw"]).all()
    assert execute(black, 3, stream="separate").outputs[channel] is NO_VALUE
    rewind = execute(black, 0)
    assert rewind.outputs[channel] is NO_VALUE
    assert rewind.outputs["status"]["model_restart"] == "video position restarted"
    repeated = execute(black, 1)
    np.testing.assert_array_equal(repeated.outputs[f"{channel}_raw"], raw)
    reset = execute(black, 2, reset=True)
    assert reset.outputs[channel] is NO_VALUE
    assert reset.outputs["status"]["generation"] != repeated.outputs["status"]["generation"]


@pytest.mark.parametrize(
    "image,params,message",
    [
        (np.zeros((8, 32), np.uint8), {}, "16–4096"),
        (np.full((32, 48), 1.5, np.float32), {}, "[0, 1]"),
        (np.zeros((32, 48), np.uint8), {"max_bytes_mb": 0.01}, "memory byte limit"),
        (np.zeros((32, 48), np.uint8), {"normalize_motion": "yes"}, "must be boolean"),
    ],
)
def test_invalid_input_and_settings_are_contained(registry, image, params, message):
    tool = registry.create(TYPE, parameters=params)
    assert tool.execute({"image": image}) is None
    assert message in tool.error
    assert not tool.memory["streams"]


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_exposure_step_and_moving_part(registry, profile):
    graph = Graph.load(EXAMPLE / "retina_preprocessing.json", registry)
    graph.set_run_profile(profile)
    timeline = {}
    details = {}
    try:
        node = graph.blocks["retina"]
        assert block_entry(node).path[:2] == ("OpenCV", "Bioinspired Vision (bioinspired)")
        for index in range(45):
            graph.run()
            result = node.last_result
            assert result and result.passed, node.error
            status = result.outputs["status"]
            assert status["timestamp_ms"] == pytest.approx(index * 50)
            assert status["frame_index"] == index
            assert result.outputs["ready"] is (index >= 5)
            if index < 5:
                assert result.outputs["detail"] is NO_VALUE
                continue
            timeline[index] = status["motion_mean"]
            assert result.payloads["detail"]["spatial"]["root_id"] == "video:image"
            if index in (11, 29):
                details[index] = result.outputs["detail"].astype(np.float32)
            if index == 31:
                motion = result.outputs["motion_raw"]
                assert motion[25:55, 56:103].mean() > 30
                assert motion[25:55, :45].mean() < 2
        assert timeline[11] < 1
        assert timeline[12] > 50
        assert timeline[29] < 0.04
        assert timeline[31] > 7
        assert np.corrcoef(details[11].ravel(), details[29].ravel())[0, 1] > 0.95
    finally:
        graph.close()
