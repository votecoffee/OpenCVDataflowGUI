import copy
import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ImagePyramid, ValuePacket, metadata, packet_for
from opencv_dataflow.toolblocks import encapsulate


def run(kind, inputs=None, parameters=None, levels=None):
    block = builtin_registry().create(kind, parameters=parameters)
    block.input_levels = levels or {}
    block.execute(inputs or {})
    assert block.state in ("pass", "fail"), block.error
    return block


def pyramid(shape=(65, 97), levels=4):
    image = ImageArray(np.random.default_rng(4).integers(0, 256, (*shape, 3), np.uint8), "rgb")
    return run("cv.image_pyramid", {"image": image}, {"levels": levels})


def test_bundled_pyramid_matches_native_odd_sizes_encoding_and_coordinates():
    block = pyramid()
    images = block.last_result.outputs["images"]
    assert isinstance(images, ImagePyramid)
    expected = block.last_inputs["image"]
    for index, image in enumerate(images.levels):
        np.testing.assert_array_equal(image, expected)
        assert image.color_format == "rgb"
        assert image.metadata["pyramid"]["level"] == index
        np.testing.assert_array_equal(
            image.metadata["pyramid"]["to_largest"], np.diag([2**index, 2**index, 1])
        )
        np.testing.assert_array_equal(
            image.metadata["spatial"]["to_root"], np.diag([2**index, 2**index, 1])
        )
        expected = cv2.pyrDown(np.asarray(expected))
    assert [list(image.shape[:2]) for image in images.levels] == [
        [65, 97],
        [33, 49],
        [17, 25],
        [9, 13],
    ]


def test_single_image_consumers_default_largest_and_select_smaller_levels():
    images = pyramid().last_result.outputs["images"]
    largest = run("cv.convert", {"image": images})
    assert largest.last_output_images["image"].shape == (65, 97)
    smaller = run("cv.convert", {"image": images}, levels={"image": "2"})
    assert smaller.last_output_images["image"].shape == (17, 25)
    assert smaller.last_result.payloads["image"]["pyramid"]["level"] == 2
    assert "source level 2" in smaller.last_result.message
    invalid = builtin_registry().create("cv.convert")
    invalid.input_levels = {"image": "8"}
    invalid.execute({"image": images})
    assert invalid.state == "error" and "no pyramid level" in invalid.error
    selected = run("cv.pyramid_level", {"images": images}, {"level": "2"})
    np.testing.assert_array_equal(selected.last_output_images["image"], images.levels[2])


@pytest.mark.parametrize(
    "kind,direction,factor", [("cv.pyramid_down", "down", 2), ("cv.pyramid_up", "up", 0.5)]
)
def test_separate_pyramid_entries_and_spatial_payload(kind, direction, factor):
    source = run("input.demo")
    output = run(kind, {"image": source.last_output_images["image"]})
    image = output.last_output_images["image"]
    expected = (cv2.pyrDown if direction == "down" else cv2.pyrUp)(
        np.asarray(source.last_output_images["image"])
    )
    np.testing.assert_array_equal(image, expected)
    np.testing.assert_array_equal(
        image.metadata["spatial"]["to_root"], np.diag([factor, factor, 1])
    )
    assert image.metadata["pyramid_operation"]["direction"] == direction


@pytest.mark.parametrize(
    "channels,split,merge,encoding",
    [
        (3, "cv.split_channels", "cv.merge_channels", "rgb"),
        (4, "cv.split_alpha_channels", "cv.merge_alpha_channels", "rgba"),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_channel_round_trip_and_payload_preservation(channels, split, merge, encoding, dtype):
    image = ImageArray(
        np.arange(10 * 12 * channels).reshape(10, 12, channels).astype(dtype),
        encoding,
        {"user": {"name": "example"}},
    )
    parts = run(split, {"image": image})
    for index, (name, plane) in enumerate(parts.last_result.outputs.items()):
        np.testing.assert_array_equal(plane, image[..., index])
        assert parts.last_result.payloads[name]["channel"]["index"] == index
        assert plane.metadata["user"] == {"name": "example"}
    merged = run(merge, parts.last_result.outputs, {"color_format": encoding})
    np.testing.assert_array_equal(merged.last_output_images["image"], image)
    assert merged.last_output_images["image"].color_format == encoding
    assert merged.last_result.payloads["image"]["user"] == {"name": "example"}
    assert "channel" not in merged.last_result.payloads["image"]


def test_extract_replace_channels_and_frame_mismatch_rejection():
    source = run("input.demo").last_output_images["image"]
    plane = run("cv.extract_channel", {"image": source}, {"channel": "1"}).last_output_images[
        "image"
    ]
    replaced = run("cv.insert_channel", {"image": source, "channel": plane}, {"channel": "0"})
    np.testing.assert_array_equal(replaced.last_output_images["image"][..., 0], source[..., 1])
    np.testing.assert_array_equal(replaced.last_output_images["image"][..., 2], source[..., 2])
    other = copy.deepcopy(plane)
    other.metadata["spatial"]["root_id"] = "different-camera"
    block = builtin_registry().create("cv.merge_channels")
    block.execute({"channel_0": plane, "channel_1": plane, "channel_2": other})
    assert block.state == "error" and "coordinate frames" in block.error


def test_generic_packets_custom_block_access_and_branch_isolation():
    class Consumer(Block):
        inputs, outputs = (Port("value", "number"),), (Port("value", "number"),)

        def process(self, inputs):
            assert self.input_payloads["value"]["units"] == "mm"
            self.input_payloads["value"]["nested"]["items"].append(2)
            return Result({"value": inputs["value"] * 2})

    packet = ValuePacket(5, {"units": "mm", "nested": {"items": [1]}})
    block = Consumer()
    block.execute({"value": packet})
    assert block.state == "pass" and block.last_result.outputs["value"] == 10
    assert packet.payload["nested"]["items"] == [1]
    assert block.last_result.payloads["value"]["units"] == "mm"
    attached = run("core.attach_payload", {"value": packet}, {"payload": {"job": "test"}})
    read = run("core.read_payload", {"value": packet_for(attached.last_result, "value")})
    assert read.last_result.outputs["payload"]["units"] == "mm"
    assert read.last_result.outputs["payload"]["job"] == "test"


def test_blob_and_region_results_flag_level_and_map_back_to_largest():
    image = np.zeros((64, 96), np.uint8)
    image[16:48, 24:72] = 255
    built = run("cv.image_pyramid", {"image": image}, {"levels": 3})
    images = built.last_result.outputs["images"]
    threshold = run("cv.threshold", {"image": images}, levels={"image": "2"})
    blobs = run(
        "inspect.components",
        {"image": threshold.last_output_images["image"]},
        {"min_area": 1, "expected_count": 1},
    )
    payload = blobs.last_result.payloads["centroids"]
    assert payload["pyramid"]["level"] == 2 and payload["spatial"]["image_size"] == [24, 16]
    read = run("core.read_payload", {"value": packet_for(blobs.last_result, "centroids")})
    mapped = run(
        "cv.transform_points",
        {
            "points": packet_for(read.last_result, "value"),
            "transform": read.last_result.outputs["transform"],
        },
    )
    np.testing.assert_allclose(
        mapped.last_result.outputs["points"], np.asarray(blobs.last_result.outputs["centroids"]) * 4
    )
    assert mapped.last_result.payloads["points"]["spatial"]["image_size"] == [96, 64]
    region = run(
        "region.shape", {"image": images}, {"rectangle": [6, 4, 12, 8]}, levels={"image": "2"}
    )
    assert region.last_result.outputs["region"]["coordinate_context"]["image_size"] == [24, 16]
    transform = run("core.read_payload", {"value": packet_for(region.last_result, "region")})
    full = run(
        "region.transform",
        {
            "image": images.largest,
            "region": region.last_result.outputs["region"],
            "transform": transform.last_result.outputs["transform"],
        },
    )
    assert full.last_result.outputs["mask"].shape == (64, 96)


def test_sets_and_level_selection_survive_nested_graph_save_and_runtime_copy(tmp_path):
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(registry.create("input.demo"))
    build = graph.add(registry.create("cv.image_pyramid"))
    gray = graph.add(registry.create("cv.convert"))
    gray.input_levels = {"image": "2"}
    graph.connect(source.id, "image", build.id, "image")
    graph.connect(build.id, "images", gray.id, "image")
    composite = encapsulate(graph, [gray.id], "Small level processing", registry)
    path = tmp_path / "pyramids.json"
    graph.save(path)
    restored = Graph.load(path, registry)
    restored.run()
    assert all(b.state == "pass" for b in restored.blocks.values())
    nested = restored.blocks[composite.id].graph.blocks[gray.id]
    assert nested.input_levels == {"image": "2"}
    assert nested.last_output_images["image"].shape == (90, 120)
    assert nested.last_result.payloads["image"]["pyramid"]["level"] == 2
    display = Graph.from_dict(restored.to_dict(), registry)
    display.blocks[composite.id].copy_results_from(restored.blocks[composite.id])
    assert (
        display.blocks[composite.id].graph.blocks[gray.id].last_result.payloads
        == nested.last_result.payloads
    )


def test_payload_added_to_bundle_reaches_selected_level_and_invalid_payloads_fail():
    built = pyramid()
    attached = run(
        "core.attach_payload",
        {"value": built.last_result.outputs["images"]},
        {"payload": {"inspection": {"recipe": "A"}}},
    )
    output = run(
        "cv.convert", {"image": packet_for(attached.last_result, "value")}, levels={"image": "1"}
    )
    assert output.last_result.payloads["image"]["inspection"] == {"recipe": "A"}
    invalid = builtin_registry().create(
        "core.attach_payload", parameters={"payload": {"spatial": {}}}
    )
    invalid.execute({"value": 1})
    assert invalid.state == "error"


def test_saved_channel_pyramid_example_preserves_payload_and_maps_centroids():
    from opencv_dataflow.examples import channel_pyramid_graph
    from opencv_dataflow.runner import describe

    registry = builtin_registry()
    graph = Graph.load(
        "examples/image_processing/channels_pyramids/channels_pyramids.json", registry
    )
    assert graph.to_dict() == channel_pyramid_graph(registry).to_dict()
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values())
        blobs = graph.blocks["pyr_components"]
        assert blobs.last_result.outputs["count"] == 2
        assert blobs.last_result.payloads["data"]["pyramid"]["level"] == 2
        assert blobs.last_result.payloads["data"]["recipe"]["revision"] == 1
        points = graph.blocks["pyr_centroids"].last_result.outputs["points"]
        np.testing.assert_allclose(sorted(points), [[90, 110], [282.5, 107.5]], atol=4)
        histogram = graph.blocks["pyr_histogram"].last_result.outputs["histogram"]
        assert histogram["channels"][0]["finite_samples"] == 380 * 220
        assert graph.blocks["pyr_preview"].last_output_images["image"].shape[:2] == (110, 190)
        receipt = next(row for row in describe(graph) if row["id"] == "pyr_components")
        assert receipt["payloads"]["data"]["spatial"]["image_size"] == [95, 55]


def test_known_same_size_wrong_frame_region_is_rejected():
    source = run("input.demo").last_output_images["image"]
    shape = run("region.shape", {"image": source})
    flipped = run("cv.flip", {"image": source}).last_output_images["image"]
    bad = builtin_registry().create("inspect.histogram")
    bad.execute({"image": flipped, "region": shape.last_result.outputs["region"]})
    assert bad.state == "error" and "coordinate frame" in bad.error


def test_bundle_depths_bounds_and_empty_coordinate_mapping():
    for dtype in (np.uint16, np.float32, np.float64):
        image = np.arange(35 * 51).reshape(35, 51).astype(dtype)
        built = run("cv.image_pyramid", {"image": image}, {"levels": 3})
        np.testing.assert_array_equal(
            built.last_result.outputs["images"].levels[1], cv2.pyrDown(image)
        )
    too_many = builtin_registry().create("cv.image_pyramid", parameters={"levels": 4})
    too_many.execute({"image": np.zeros((2, 2), np.uint8)})
    assert too_many.state == "error" and "more levels" in too_many.error
    with pytest.raises(ValueError, match="payloads"):
        ImagePyramid((np.zeros((10, 10), np.uint8),))
    images = pyramid().last_result.outputs["images"]
    context = metadata(images.levels[2])
    read = run("core.read_payload", {"value": ValuePacket([], context)})
    mapped = run(
        "cv.transform_points",
        {"points": ValuePacket([], context), "transform": read.last_result.outputs["transform"]},
    )
    assert mapped.last_result.outputs["points"] == []
    assert mapped.last_result.payloads["points"]["spatial"]["image_size"] == [97, 65]


def test_custom_geometry_composes_tagged_arrays_and_explicit_packets():
    class Crop(Block):
        inputs = (Port("image"),)
        outputs = (Port("image"), Port("geometry", "data"))

        def process(self, inputs):
            return Result(
                {
                    "image": inputs["image"][5:15, 7:27].copy(),
                    "geometry": {
                        "output_to_input": [[1, 0, 7], [0, 1, 5], [0, 0, 1]],
                        "input_size": [480, 360],
                        "output_size": [20, 10],
                    },
                }
            )

    image = run("input.demo").last_output_images["image"]
    crop = Crop()
    crop.execute({"image": image})
    assert crop.state == "pass", crop.error
    np.testing.assert_array_equal(
        crop.last_result.payloads["image"]["spatial"]["to_root"], [[1, 0, 7], [0, 1, 5], [0, 0, 1]]
    )

    class Producer(Block):
        outputs = (Port("value", "number"),)

        def process(self, inputs):
            return Result({"value": ValuePacket(3, {"units": "mm"})})

    producer = Producer()
    producer.execute({})
    assert producer.state == "pass"
    assert producer.last_result.outputs["value"] == 3
    assert producer.last_result.payloads["value"] == {"units": "mm"}


def test_headless_pyramid_receipt_serializes_frames_and_json_primary_values(tmp_path):
    from opencv_dataflow.runner import main

    path = tmp_path / "pyramids.json"
    assert (
        main(
            [
                "examples/image_processing/channels_pyramids/channels_pyramids.json",
                "--repeat",
                "2",
                "--report",
                str(path),
            ]
        )
        == 0
    )
    receipt = json.loads(path.read_text())
    assert receipt["passed"] and len(receipt["runs"]) == 2
    rows = {row["id"]: row for row in receipt["runs"][0]["blocks"]}
    assert rows["pyr_coordinates"]["outputs"]["value"]
    assert "images" not in rows["pyr_set"]["outputs"]
    assert rows["pyr_components"]["payloads"]["data"]["pyramid"]["level"] == 2
    assert rows["pyr_centroids"]["payloads"]["points"]["spatial"]["image_size"] == [380, 220]


def test_image_payload_mutation_is_private_to_consumer_snapshot():
    class Mutate(Block):
        inputs, outputs = (Port("image"),), (Port("image"),)

        def process(self, inputs):
            inputs["image"].metadata["recipe"]["items"].append(2)
            return Result({"image": inputs["image"]})

    original = ImageArray(np.zeros((10, 20), np.uint8), "gray", {"recipe": {"items": [1]}})
    block = Mutate()
    block.execute({"image": original})
    assert block.state == "pass"
    assert original.metadata["recipe"]["items"] == [1]
    assert block.last_inputs["image"].metadata["recipe"]["items"] == [1]
    assert block.last_result.payloads["image"]["recipe"]["items"] == [1, 2]
