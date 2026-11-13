"""Native animation parameters, independent frame truth, timing and safe failure behavior."""

import copy
import io
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from opencv_dataflow.animation_blocks import (
    ANIMATION_BLOCKS,
    AnimationFile,
    DecodeAnimation,
    EncodeAnimation,
    WriteAnimationFile,
)
from opencv_dataflow.animation_codec import (
    GIF_DITHER,
    GIF_QUALITY,
    PNG_STRATEGY,
    EncodedAnimation,
    header,
)
from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame, packet_for
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key


@pytest.fixture
def group():
    y, x = np.indices((32, 48))
    images = []
    clock = TicketClock()
    for index in range(3):
        image = np.dstack(
            (
                (x * 3 + index * 41) % 256,
                (y * 5 + index * 13) % 256,
                np.full_like(x, 50 + index * 70),
            )
        ).astype(np.uint8)
        image[8:18, 5 + index * 9 : 12 + index * 9] = 255
        payload = {
            "ticket": clock.issue("camera"),
            "spatial": frame([48, 32], f"camera-{index}"),
            "acquisition": {"encoder": index * 17},
            "animation_frame": {"duration_ms": [80, 120, 200][index], "loop_count": 3},
        }
        images.append(ValuePacket(ImageArray(image, "bgr", payload), payload))
    return FlowGroup(tuple(images), ("first", "second", "third"))


def successful(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    assert block.state == "pass", block.error
    return result


@pytest.mark.parametrize("codec", ["apng", "webp", "gif"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_buffer_roundtrip_order_timing_contexts_and_independent_pixels(group, codec, profile):
    before = copy.deepcopy(group)
    encoder = EncodeAnimation(parameters={"codec": codec})
    graph = Graph()
    graph.add(encoder)
    graph.set_run_profile(profile)
    encoded = successful(encoder, {"frames": group})
    decoder = DecodeAnimation()
    decoded = successful(decoder, {"encoded": packet_for(encoded, "encoded")})
    details = decoded.outputs["info"]
    assert decoded.outputs["frames"].labels == group.labels
    assert details["durations_ms"] == [80, 120, 200]
    assert details["loop_count"] == 3 and details["frame_count"] == 3
    for source, restored, saved in zip(
        group.items, decoded.outputs["frames"].items, before.items, strict=True
    ):
        assert ticket_key(source.payload["ticket"]) == ticket_key(restored.payload["ticket"])
        assert restored.payload["spatial"] == source.payload["spatial"]
        assert restored.payload["acquisition"] == source.payload["acquisition"]
        np.testing.assert_array_equal(source.value, saved.value)
        if codec != "gif":
            np.testing.assert_array_equal(source.value, restored.value)
    data = encoded.outputs["encoded"].buffer
    with Image.open(io.BytesIO(data)) as independent:
        assert independent.n_frames == 3
        for i, packet in enumerate(decoded.outputs["frames"].items):
            independent.seek(i)
            rgb = np.asarray(independent.convert("RGB"))
            np.testing.assert_array_equal(rgb, np.asarray(packet.value)[..., :3][..., ::-1])
            assert independent.info["duration"] == [80, 120, 200][i]
    assert encoded.outputs["info"]["exact_visible_samples"] == (codec != "gif")
    json.dumps(encoded.outputs["encoded"].describe(), allow_nan=False)
    graph.close()


@pytest.mark.parametrize("codec,extension", [("apng", ".apng"), ("webp", ".webp"), ("gif", ".gif")])
def test_native_file_apis_unicode_paths_and_exclusive_output(group, tmp_path, codec, extension):
    path = tmp_path / ("s\u00e9quence-\u90e8\u54c1" + extension)
    writer = WriteAnimationFile(parameters={"codec": codec, "path": str(path)})
    result = successful(writer, {"frames": group})
    frozen = path.read_bytes()
    assert result.outputs["frame_count"] == 3 and result.outputs["byte_count"] == len(frozen)
    source = AnimationFile(parameters={"path": str(path)})
    decoded = successful(source, {})
    assert decoded.outputs["durations"] == [80, 120, 200]
    assert decoded.outputs["ticket"]["value"] == path.name
    assert all(
        p.payload["ticket"] == decoded.outputs["ticket"] for p in decoded.outputs["frames"].items
    )
    assert len({p.payload["spatial"]["root_id"] for p in decoded.outputs["frames"].items}) == 3
    assert writer.execute({"frames": group}) is None and "preserved" in writer.error
    assert path.read_bytes() == frozen and list(tmp_path.iterdir()) == [path]


def test_subset_preview_formatting_graphics_and_reencoding_payload_durations(group):
    original = successful(EncodeAnimation(), {"frames": group})
    block = DecodeAnimation(
        parameters={
            "start_frame": 1,
            "frame_count": 2,
            "preview_frame": 1,
            "color_format": "gray",
            "pixel_depth": "uint16",
        }
    )
    subset = successful(block, {"encoded": original.outputs["encoded"]})
    assert subset.outputs["durations"] == [120, 200]
    assert subset.outputs["image"].dtype == np.uint16 and subset.outputs["image"].ndim == 2
    assert [p.payload["animation_frame"]["index"] for p in subset.outputs["frames"].items] == [1, 2]
    assert len(subset.drawings["animation frame"].objects) == 1
    block.graphics_mode = "off"
    no_graphics = successful(block, {"encoded": original.outputs["encoded"]})
    assert not no_graphics.drawings
    np.testing.assert_array_equal(no_graphics.outputs["image"], subset.outputs["image"])
    full = successful(DecodeAnimation(), {"encoded": original.outputs["encoded"]})
    again = successful(EncodeAnimation(), {"frames": full.outputs["frames"]})
    assert again.outputs["info"]["durations_ms"] == [80, 120, 200]


@pytest.mark.parametrize("layout", ["gray", "rgb", "rgba"])
@pytest.mark.parametrize("codec", ["apng", "webp"])
def test_color_layout_and_alpha_roundtrips(group, layout, codec):
    packets = []
    for item in group.items:
        image = np.asarray(item.value)
        if layout == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = image[..., ::-1].copy()
        if layout == "rgba":
            alpha = np.full(image.shape[:2], 173, np.uint8)
            alpha[:3] = 0
            image = np.dstack((image, alpha))
        packets.append(ValuePacket(ImageArray(image, layout), {}))
    converted = FlowGroup(tuple(packets), group.labels)
    encoded = successful(EncodeAnimation(parameters={"codec": codec}), {"frames": converted})
    decoded = successful(DecodeAnimation(), {"encoded": encoded.outputs["encoded"]})
    for source, actual in zip(converted.items, decoded.outputs["frames"].items, strict=True):
        expected = np.asarray(source.value)
        observed = np.asarray(actual.value)
        if layout == "gray":
            if observed.ndim == 3:
                observed = observed[..., 0]
            np.testing.assert_array_equal(observed, expected)
        elif layout == "rgb":
            np.testing.assert_array_equal(observed, expected)
        else:
            np.testing.assert_array_equal(observed[..., 3], expected[..., 3])
            visible = expected[..., 3] != 0
            np.testing.assert_array_equal(observed[visible, :3], expected[visible, :3])


@pytest.mark.parametrize("quality", list(GIF_QUALITY))
def test_gif_quality_dropdowns_and_explicit_quantization(group, quality):
    p = {
        "codec": "gif",
        "gif_quality": quality,
        "gif_timing": "floor_10ms",
        "duration_mode": "explicit",
        "durations_ms": [83, 129, 208],
    }
    result = successful(EncodeAnimation(parameters=p), {"frames": group})
    assert result.outputs["info"]["requested_durations_ms"] == [83, 129, 208]
    assert result.outputs["info"]["durations_ms"] == [80, 120, 200]


@pytest.mark.parametrize("dither", list(GIF_DITHER))
def test_gif_dither_and_local_palette_controls(group, dither):
    result = successful(
        EncodeAnimation(
            parameters={"codec": "gif", "gif_dither": dither, "gif_color_table": "local"}
        ),
        {"frames": group},
    )
    assert header(result.outputs["encoded"].buffer)["codec"] == "gif"


@pytest.mark.parametrize("strategy", ["native", *PNG_STRATEGY])
def test_apng_compression_controls_preserve_pixels(group, strategy):
    result = successful(
        EncodeAnimation(parameters={"png_compression": 9, "png_strategy": strategy}),
        {"frames": group},
    )
    assert result.outputs["info"]["maximum_sample_errors"] == [0, 0, 0]


def test_webp_background_channel_order_and_lossy_quality(group):
    background = [12, 23, 34, 255]
    result = successful(
        EncodeAnimation(
            parameters={
                "codec": "webp",
                "background_bgra": background,
                "webp_mode": "lossy",
                "webp_quality": 42,
            }
        ),
        {"frames": group},
    )
    with Image.open(io.BytesIO(result.outputs["encoded"].buffer)) as independent:
        assert independent.info["background"] == (34, 23, 12, 255)
    assert result.outputs["info"]["background_bgra"] == background
    assert max(result.outputs["info"]["maximum_sample_errors"]) > 0


@pytest.mark.parametrize("layout", ["gray", "bgra"])
def test_apng_poster_is_separate_from_the_timed_sequence(group, layout):
    code = cv2.COLOR_BGR2GRAY if layout == "gray" else cv2.COLOR_BGR2BGRA
    group = FlowGroup(
        tuple(ValuePacket(cv2.cvtColor(np.asarray(p.value), code), p.payload) for p in group.items),
        group.labels,
    )
    poster = np.full_like(group.items[0].value, 71)
    if layout == "bgra":
        poster[..., 3] = 255
    encoded = successful(EncodeAnimation(), {"frames": group, "still_image": poster})
    decoded = successful(DecodeAnimation(), {"encoded": encoded.outputs["encoded"]})
    assert decoded.outputs["frame_count"] == 3 and decoded.outputs["info"]["has_still_image"]
    np.testing.assert_array_equal(decoded.outputs["still_image"], poster)
    with Image.open(io.BytesIO(encoded.outputs["encoded"].buffer)) as independent:
        assert independent.info["default_image"] and independent.n_frames == 4
        independent.seek(1)
        expected = np.asarray(group.items[0].value)
        if layout == "gray":
            expected = cv2.cvtColor(expected, cv2.COLOR_GRAY2BGR)
        else:
            expected = expected[..., :3]
        np.testing.assert_array_equal(np.array(independent.convert("RGB")), expected[..., ::-1])


def test_unreliable_native_rgb_poster_is_rejected(group):
    block = EncodeAnimation()
    assert (
        block.execute({"frames": group, "still_image": np.zeros_like(group.items[0].value)}) is None
    )
    assert "RGB posters" in block.error and not block.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"codec": "gif", "duration_mode": "explicit", "durations_ms": [81, 120, 200]},
        {"duration_mode": "explicit", "durations_ms": [100]},
        {"duration_mode": "explicit", "durations_ms": [100, 0, 200]},
        {"duration_mode": "explicit", "durations_ms": [100, True, 200]},
        {"maximum_pixels": 1},
        {"background_bgra": [0, 0]},
        {"background_bgra": [0, 0, 0, 255]},
        {"codec": "gif", "background_bgra": [1, 0, 0, 0]},
        {"codec": "no-such-codec"},
        {"loop_count": -1},
        {"webp_quality": 101},
        {"png_compression": 10},
    ],
)
def test_invalid_parameters_emit_nothing_and_contain_failure(group, parameters):
    block = EncodeAnimation(parameters=parameters)
    assert block.execute({"frames": group}) is None
    assert block.state == "error" and block.error_kind == "invalid"
    assert not block.current_emissions and not block.last_output_images


@pytest.mark.parametrize("codec", ["apng", "webp"])
def test_native_identical_frame_merging_is_detected_before_publication(group, codec, tmp_path):
    same = FlowGroup(tuple(ValuePacket(group.items[0].value, {}) for _ in range(3)), group.labels)
    block = EncodeAnimation(parameters={"codec": codec})
    assert block.execute({"frames": same}) is None and "frame count" in block.error
    path = tmp_path / ("merged.png" if codec == "apng" else "merged.webp")
    writer = WriteAnimationFile(parameters={"codec": codec, "path": str(path)})
    assert writer.execute({"frames": same}) is None
    assert not path.exists() and not list(tmp_path.iterdir())


@pytest.mark.parametrize("data", [b"", b"not an animation", b"GIF89a" + b"\xff" * 20])
def test_malformed_buffers_are_contained(data):
    block = DecodeAnimation()
    assert block.execute({"encoded": data}) is None and block.state == "error"
    assert not block.current_emissions


def test_pixel_limit_is_checked_before_native_decoder(group, monkeypatch):
    encoded = successful(EncodeAnimation(), {"frames": group})
    calls = []
    monkeypatch.setattr(cv2, "imdecodeanimation", lambda *a, **k: calls.append(True))
    block = DecodeAnimation(parameters={"maximum_pixels": 1})
    assert block.execute({"encoded": encoded.outputs["encoded"]}) is None
    assert not calls and "budget" in block.error


@pytest.mark.parametrize("start,count,preview", [(3, 0, 0), (0, 1, 1)])
def test_invalid_selection_and_preview_are_reported(group, start, count, preview):
    encoded = successful(EncodeAnimation(), {"frames": group})
    block = DecodeAnimation(
        parameters={"start_frame": start, "frame_count": count, "preview_frame": preview}
    )
    assert block.execute({"encoded": encoded.outputs["encoded"]}) is None
    assert block.error_kind == "invalid"


def test_buffer_inputs_and_packet_metadata_validation(group):
    encoded = successful(EncodeAnimation(), {"frames": group}).outputs["encoded"]
    for raw in (
        encoded.buffer,
        np.frombuffer(encoded.buffer, np.uint8),
        np.frombuffer(encoded.buffer, np.uint8).reshape(-1, 1),
    ):
        assert successful(DecodeAnimation(), {"encoded": raw}).outputs["frame_count"] == 3
    with pytest.raises(ValueError):
        EncodedAnimation("gif", encoded.buffer)
    with pytest.raises(ValueError):
        EncodedAnimation("apng", encoded.buffer, ({},))


def test_all_parameters_persist_in_xml_and_graph_roundtrip(tmp_path):
    registry = builtin_registry()
    graph = Graph()
    for cls in ANIMATION_BLOCKS:
        block = graph.add(registry.create(cls.type_id))
        path = tmp_path / ("settings." + cls.type_id + ".xml")
        save_block_xml(block, path)
        assert read_block_xml(path)["parameters"] == block.parameters
    graph.save(tmp_path / "animation.json")
    restored = Graph.load(tmp_path / "animation.json", registry)
    assert restored.to_dict() == graph.to_dict()
    restored.close()
    graph.close()


@pytest.mark.parametrize("threshold", [0, 1, 128, 255])
def test_gif_alpha_threshold_matches_independent_decoder(group, threshold):
    packets = []
    for item in group.items:
        alpha = np.tile(np.array([0, 127, 128, 255], np.uint8), (32, 12))
        packets.append(ValuePacket(np.dstack((item.value, alpha)), item.payload))
    converted = FlowGroup(tuple(packets), group.labels)
    encoded = successful(
        EncodeAnimation(parameters={"codec": "gif", "gif_transparency_threshold": threshold}),
        {"frames": converted},
    )
    decoded = successful(DecodeAnimation(), {"encoded": encoded.outputs["encoded"]})
    with Image.open(io.BytesIO(encoded.outputs["encoded"].buffer)) as independent:
        for i, packet in enumerate(decoded.outputs["frames"].items):
            independent.seek(i)
            actual = np.asarray(packet.value)
            if actual.shape[2] == 3:
                actual = np.dstack((actual, np.full(actual.shape[:2], 255, np.uint8)))
            np.testing.assert_array_equal(
                actual[..., 3], np.array(independent.convert("RGBA"))[..., 3]
            )
            expected = np.where(alpha < threshold, 0, 255).astype(np.uint8)
            np.testing.assert_array_equal(actual[..., 3], expected)


def test_background_and_wired_durations_roundtrip(group):
    first = successful(
        EncodeAnimation(
            parameters={
                "codec": "webp",
                "background_bgra": [12, 23, 34, 255],
                "loop_mode": "override",
                "loop_count": 6,
            }
        ),
        {"frames": group, "durations": [91, 133, 211]},
    )
    decoded = successful(DecodeAnimation(), {"encoded": first.outputs["encoded"]})
    second = successful(
        EncodeAnimation(parameters={"codec": "webp"}), {"frames": decoded.outputs["frames"]}
    )
    assert second.outputs["info"]["durations_ms"] == [91, 133, 211]
    assert second.outputs["info"]["loop_count"] == 6
    assert second.outputs["info"]["background_bgra"] == [12, 23, 34, 255]


@pytest.mark.parametrize("bad", ["depth", "shape", "not_image", "one_frame"])
def test_incompatible_frame_collections_are_contained(group, bad):
    packets = list(copy.deepcopy(group.items))
    if bad == "depth":
        packets[0].value = np.asarray(packets[0].value).astype(np.uint16)
    elif bad == "shape":
        packets[0].value = np.zeros((1, 1, 3), np.uint8)
    elif bad == "not_image":
        packets[0].value = {"not": "an image"}
    else:
        packets = packets[:1]
    bad_group = FlowGroup(tuple(packets), group.labels[: len(packets)])
    block = EncodeAnimation()
    assert block.execute({"frames": bad_group}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


def test_destination_created_during_native_write_is_preserved(group, tmp_path, monkeypatch):
    target = tmp_path / "race.png"
    writer = WriteAnimationFile(parameters={"path": str(target)})
    normal = writer.encoded_result

    def competing_writer(*args, **kwargs):
        result = normal(*args, **kwargs)
        target.write_bytes(b"another owner's output")
        return result

    monkeypatch.setattr(writer, "encoded_result", competing_writer)
    assert writer.execute({"frames": group}) is None
    assert target.read_bytes() == b"another owner's output"
    assert list(tmp_path.iterdir()) == [target]
