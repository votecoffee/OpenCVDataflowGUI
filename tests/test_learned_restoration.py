"""Independent convolution references, same-grid metadata and bounded native contracts."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.learned_restoration import RestoreImage
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/learned_restoration"


def tool(case="whole", **parameters):
    block = RestoreImage(
        parameters={"model_path": str(BASE / "models" / case / "model.onnx"), **parameters}
    )
    return block


def source(encoding="gray", depth="uint8"):
    yy, xx = np.indices((23, 31))
    values = ((xx * 17 + yy * 31) % 256).astype(np.uint8)
    if encoding != "gray":
        values = np.dstack((values, values // 2, 255 - values))
    if encoding in ("rgba", "bgra"):
        values = np.dstack((values, (xx * 7).astype(np.uint8)))
    if depth == "uint16":
        values = values.astype(np.uint16) * 257
    elif depth == "float32":
        values = values.astype(np.float32) / 255
    return ImageArray(values, encoding)


def reference(image, mean, std, blend=1):
    maximum = 1 if image.dtype == np.float32 else np.iinfo(image.dtype).max
    unit = np.asarray(image).astype(np.float32) / maximum
    planes = unit[..., None] if unit.ndim == 2 else unit[..., :3]
    mean, std = np.broadcast_to(mean, (planes.shape[2],)), np.broadcast_to(std, (planes.shape[2],))
    restored = []
    for index in range(planes.shape[2]):
        prepared = (planes[..., index] - mean[index]) / std[index]
        # Native-independent image-filter reference for the arithmetic ONNX graph.
        average = cv2.filter2D(
            prepared, -1, np.full((3, 3), 1 / 9, np.float32), borderType=cv2.BORDER_CONSTANT
        )
        restored.append(
            planes[..., index] + blend * (average * std[index] + mean[index] - planes[..., index])
        )
    values = np.clip(np.stack(restored, axis=-1), 0, 1)
    if image.dtype != np.float32:
        values = np.rint(values * maximum).astype(image.dtype)
    result = np.asarray(image).copy()
    if image.ndim == 2:
        result = values[..., 0]
    elif image.shape[2] == 4:
        result[..., :3] = values
    else:
        result = values
    return result


@pytest.mark.parametrize("encoding", ["gray", "bgr", "rgb", "bgra", "rgba"])
@pytest.mark.parametrize("depth", ["uint8", "uint16", "float32"])
def test_independent_filter_reference_depth_color_alpha_and_no_source_mutation(encoding, depth):
    image = source(encoding, depth)
    before = image.copy()
    color = "gray" if encoding == "gray" else "rgb"
    block = tool("whole" if encoding == "gray" else "rgb_alpha", color_format=color, blend=0.65)
    try:
        result = block.execute({"image": image})
        assert result is not None, block.error
        expected = reference(image, [0], [1], 0.65)
        np.testing.assert_allclose(
            result.outputs["image"], expected, atol=2e-6 if depth == "float32" else 1
        )
        assert result.outputs["image"].dtype == image.dtype
        assert result.outputs["image"].color_format == encoding
        if encoding in ("bgra", "rgba"):
            np.testing.assert_array_equal(result.outputs["image"][..., 3], image[..., 3])
        np.testing.assert_array_equal(image, before)
    finally:
        block.close()


@pytest.mark.parametrize(
    "case,changes",
    [
        ("whole", {}),
        ("residual", {"output_mode": "noise_residual"}),
        ("nhwc", {"input_layout": "NHWC", "output_layout": "NHWC"}),
    ],
)
def test_layout_residual_and_normalization_units_have_an_independent_reference(case, changes):
    image = source(depth="float32")
    block = tool(case, mean=[0.25], std=[0.5], **changes)
    try:
        result = block.execute({"image": image})
        assert result is not None, block.error
        np.testing.assert_allclose(
            result.outputs["image"], reference(image, [0.25], [0.5]), atol=2e-6
        )
    finally:
        block.close()


def test_context_tiles_uneven_borders_selection_and_same_grid_payload():
    image = source(depth="uint16")
    context = {
        "spatial": frame([31, 23], "camera"),
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "part", "revision": 1, "digest": "a" * 64},
        "aux": {"batch": 12},
        "pyramid": {
            "level": 1,
            "largest_size": [62, 46],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
    }
    mask = np.zeros((23, 31), np.uint8)
    mask[2:18, 4:27] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [31, 23],
        "shape": "rectangle",
        "rectangle": [8, 6, 20, 15],
    }
    block = tool(processing_mode="halo_tiles", tile_width=9, tile_height=7, halo=1)
    try:
        result = block.execute(
            {"image": ValuePacket(image, context), "mask": mask, "region": region}
        )
        assert result is not None, block.error
        selected = np.zeros((23, 31), bool)
        selected[6:18, 8:27] = True
        expected = image.copy()
        full = reference(image, [0], [1])
        expected[selected] = full[selected]
        np.testing.assert_allclose(result.outputs["image"], expected, atol=1)
        np.testing.assert_array_equal(result.outputs["image"][~selected], image[~selected])
        np.testing.assert_array_equal(
            result.outputs["processed_mask"], selected.astype(np.uint8) * 255
        )
        for payload in result.payloads.values():
            for name, value in context.items():
                assert payload[name] == value
        calls = result.outputs["receipt"]["calls"]
        assert len(calls) == result.outputs["info"]["native_calls"]
        assert all("restoration_window" in call for call in calls)
    finally:
        block.close()


@pytest.mark.parametrize("blank", ["selection", "blend"])
def test_noop_emits_current_input_without_model_io(blank, monkeypatch):
    import opencv_dataflow.learned_restoration as module

    monkeypatch.setattr(module, "model_bytes", lambda *a, **k: pytest.fail("model I/O"))
    block = RestoreImage(parameters={"blend": 0 if blank == "blend" else 1})
    args = {"image": source()}
    if blank == "selection":
        args["mask"] = np.zeros((23, 31), np.uint8)
    result = block.execute(args)
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], args["image"])
    assert result.outputs["receipt"] is NO_VALUE and result.outputs["info"]["native_calls"] == 0
    block.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"halo": -1},
        {"tile_width": True},
        {"tile_height": 0},
        {"maximum_tiles": 0},
        {"blend": True},
        {"blend": float("nan")},
        {"blend": 1.1},
        {"std": [0]},
        {"mean": [True]},
        {"mean": [0, 0]},
        {"output_names": ["a", "b"]},
        {"input_name": 12},
        {"fusion": "yes"},
        {"output_layout": "bad"},
        {"backend": "cuda", "target": "cpu"},
        {"target": "opencl", "engine": "new"},
        {"fusion": False, "engine": "auto"},
    ],
)
def test_parameter_failures_contained_and_recover(changes):
    block = tool(**changes)
    try:
        assert block.execute({"image": source()}) is None
        assert block.error
        block.parameters.update(copy.deepcopy(tool().parameters))
        assert block.execute({"image": source()}) is not None, block.error
    finally:
        block.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_source_pixels": 100},
        {"maximum_processed_pixels": 100},
        {"max_output_elements": 100},
        {"processing_mode": "halo_tiles", "tile_width": 5, "tile_height": 5, "maximum_tiles": 1},
    ],
)
def test_work_budget_preflight_precedes_all_model_io(changes, monkeypatch):
    import opencv_dataflow.learned_restoration as module

    monkeypatch.setattr(module, "model_bytes", lambda *a, **k: pytest.fail("model I/O"))
    block = tool(**changes)
    try:
        assert block.execute({"image": source()}) is None and block.error
    finally:
        block.close()


@pytest.mark.parametrize(
    "image",
    [
        source("rgb"),
        DepthArray(np.ones((8, 8))),
        ImageArray(np.zeros((8, 8, 3), np.uint8), "gray"),
        ImageArray(np.full((8, 8), 1.2, np.float32), "gray"),
        ImageArray(np.full((8, 8), np.nan, np.float32), "gray"),
    ],
)
def test_invalid_source_and_model_channel_contracts_reject(image):
    block = tool()
    try:
        assert block.execute({"image": image}) is None and block.error
    finally:
        block.close()


def test_sealed_snapshot_is_atomic_across_tiles_and_tampering_recovers(tmp_path):
    import shutil

    shutil.copytree(BASE / "models/tiles", tmp_path / "bundle")
    block = RestoreImage(
        parameters={
            "contract_path": "bundle/contract.json",
            "tile_width": 32,
            "tile_height": 28,
            "halo": 1,
            "processing_mode": "halo_tiles",
        }
    )
    block.project_dir = tmp_path
    model = tmp_path / "bundle/model.onnx"
    raw = model.read_bytes()
    original = block.infer_tensors
    count = [0]

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        count[0] += 1
        if count[0] == 1:
            model.write_bytes(b"x" * len(raw))
        return result

    block.infer_tensors = mutate
    try:
        image = ImageArray(np.tile(np.arange(73, dtype=np.uint8), (61, 1)), "gray")
        first = block.execute({"image": image})
        assert first is not None and count[0] > 1, block.error
        assert block.execute({"image": image}) is None and "checksum" in block.error
        assert block._net is None
        model.write_bytes(raw)
        assert block.execute({"image": image}) is not None, block.error
        block.parameters["halo"] = 2
        assert block.execute({"image": image}) is None and block.error
    finally:
        block.close()


def test_native_import_and_wrong_grid_failure_recover_without_events(tmp_path):
    import onnx
    from onnx import helper

    path = tmp_path / "model.onnx"
    path.write_bytes(b"malformed ONNX")
    block = tool(model_path=str(path))
    try:
        assert block.execute({"image": source()}) is None and block.error
        assert block._net is None
        model = onnx.load(BASE / "models/whole/model.onnx")
        model.graph.node.append(
            helper.make_node(
                "AveragePool", ["output"], ["smaller"], kernel_shape=[2, 2], strides=[2, 2]
            )
        )
        model.graph.output[0].name = "smaller"
        for dim in model.graph.output[0].type.tensor_type.shape.dim[2:]:
            dim.ClearField("dim_param")
        onnx.save(model, path)
        assert block.execute({"image": source()}) is None
        assert "patch grid" in block.error and block._net is None
        block.parameters["model_path"] = str(BASE / "models/whole/model.onnx")
        assert block.execute({"image": source()}) is not None, block.error
    finally:
        block.close()
