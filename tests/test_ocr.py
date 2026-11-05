"""Exercise OpenCV OCR with deterministic ONNX fixtures and real graph wiring."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def make_model(path, input_shape, output_shape, nodes, initializers=()):
    onnx = pytest.importorskip("onnx")
    helper, types = onnx.helper, onnx.TensorProto
    graph = helper.make_graph(
        [helper.make_node(*args, **attrs) for args, attrs in nodes],
        "ocr_fixture",
        [helper.make_tensor_value_info("image", types.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", types.FLOAT, output_shape)],
        [
            helper.make_tensor(name, getattr(types, kind), shape, data)
            for name, kind, shape, data in initializers
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())
    return str(path)


def test_probability_map_adapter_restores_letterboxed_image_coordinates(tmp_path):
    model = make_model(
        tmp_path / "map.onnx",
        [1, 1, 64, 64],
        [1, 1, 64, 64],
        [(("Relu", ["image"], ["output"]), {})],
    )
    registry = builtin_registry()
    graph = Graph()
    image = np.zeros((64, 128), np.uint8)
    image[20:40, 24:104] = 255
    dnn = graph.add(
        registry.create(
            "ml.dnn",
            parameters={
                "model_path": model,
                "width": 64,
                "height": 64,
                "color_format": "gray",
                "resize_mode": "letterbox",
                "mean": [0.0],
                "std": [1.0],
            },
        )
    )
    regions = graph.add(registry.create("ml.text_regions", parameters={"padding_ratio": 1}))
    for output in ("tensors", "preprocessing"):
        graph.connect(dnn.id, output, regions.id, output)
    try:
        graph.run(injected={(dnn.id, "image"): image, (regions.id, "image"): image})
        result = regions.last_result
        assert result is not None, regions.error
        assert result.outputs["count"] == 1
        record = result.outputs["regions"]["items"][0]
        np.testing.assert_allclose(record["center"], [63.5, 29.5], atol=1)
        np.testing.assert_allclose(
            record["corners"], [[24, 20], [103, 20], [103, 39], [24, 39]], atol=2
        )
        assert record["confidence"] == pytest.approx(1)
    finally:
        graph.close()


@pytest.mark.parametrize("decoder", ["CTC-greedy", "CTC-prefix-beam-search"])
def test_ocr_decodes_repeat_blank_and_rectified_region(tmp_path, decoder):
    model = make_model(
        tmp_path / "ctc.onnx",
        [1, 1, 3, 5],
        [5, 1, 3],
        [
            (("Conv", ["image", "weights", "bias"], ["logits"]), {"kernel_shape": [1, 1]}),
            (("ReduceMean", ["logits"], ["reduced"]), {"axes": [2], "keepdims": 1}),
            (("Softmax", ["reduced"], ["prob"]), {"axis": 1}),
            (("Transpose", ["prob"], ["ordered"]), {"perm": [3, 0, 1, 2]}),
            (("Reshape", ["ordered", "shape"], ["output"]), {}),
        ],
        [
            ("weights", "FLOAT", [3, 1, 1, 1], [0, 20, -20]),
            ("bias", "FLOAT", [3], [13, 0, 20]),
            ("shape", "INT64", [3], [5, 1, 3]),
        ],
    )
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("A\nB\n", encoding="utf-8")
    block = builtin_registry().create(
        "ml.recognize_text",
        parameters={
            "model_path": model,
            "vocabulary_path": str(vocabulary),
            "width": 5,
            "height": 3,
            "decoder": decoder,
        },
    )
    tile = np.tile(np.array([255, 255, 128, 0, 255], np.uint8), (3, 1))
    direct = block.execute({"image": tile})
    assert direct is not None, block.error
    assert direct.outputs["text"] == "ABA"
    scene = np.zeros((20, 20), np.uint8)
    scene[4:7, 9:14] = tile
    corners = [[9, 4], [13, 4], [13, 6], [9, 6]]
    regions = {"model": "ocr_regions_v1", "items": [{"corners": corners, "confidence": 0.9}]}
    result = block.execute({"image": scene, "regions": regions})
    assert result is not None, block.error
    assert result.outputs["text"] == "ABA"
    assert result.outputs["reads"][0]["corners"] == corners
    no_read = block.execute({"image": tile, "mask": np.zeros_like(tile)})
    assert not no_read.passed
    assert no_read.fail_reasons[0]["code"] == "ocr_read_count_below_minimum"


def test_verify_text_reads_qr_output_and_rejects_unexpected_serial():
    registry = builtin_registry()
    graph = Graph()
    reader = graph.add(registry.create("inspect.read_qr", parameters={"mode": "single"}))
    verifier = graph.add(
        registry.create(
            "inspect.verify_text",
            parameters={
                "comparison": "exact",
                "expected": "lot123",
                "character_set": "ascii_alphanumeric",
                "case_sensitive": False,
                "minimum_length": 6,
                "maximum_length": 6,
            },
        )
    )
    graph.connect(reader.id, "codes", verifier.id, "text")
    try:
        for expected, text in [(True, "LOT123"), (False, "LOT12Z")]:
            qr = cv2.QRCodeEncoder_create().encode(text)
            image = np.pad(np.repeat(np.repeat(qr, 8, axis=0), 8, axis=1), 24, constant_values=255)
            graph.run(injected={(reader.id, "image"): image})
            result = verifier.last_result
            assert result is not None, verifier.error
            assert result.passed is expected
            if not expected:
                assert result.outputs["checks"][0]["reasons"] == ["text_expectation_mismatch"]
    finally:
        graph.close()


def test_verify_text_character_rules_and_aggregate_policy():
    block = builtin_registry().create(
        "inspect.verify_text", parameters={"character_set": "ascii_digits", "aggregate": "any"}
    )
    inputs = {"text": [{"text": " 0012 "}, {"text": "٠١"}, {"text": ""}]}
    result = block.execute(inputs)
    assert result.passed
    assert result.outputs["matches"] == ["0012"]
    assert result.outputs["checks"][1]["reasons"] == ["text_disallowed_character"]
    assert result.outputs["checks"][2]["reasons"] == ["text_too_short"]
    block.parameters["aggregate"] = "all"
    assert not block.execute(inputs).passed
    assert block.execute({"text": []}).fail_reasons[0]["code"] == "no_text_reads"
