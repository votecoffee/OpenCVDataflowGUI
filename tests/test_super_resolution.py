"""Pinned native pixels, quantization/selection contracts, cache integrity and cleanup."""

import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.images import DepthArray, ImageArray, convert_depth, opencv_image
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.super_resolution import SuperResolution
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/super_resolution"
MODEL = BASE / "assets/FSRCNN_x2.pb"
DIGEST = "366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c"


def tool(**parameters):
    return SuperResolution(
        parameters={"model_path": str(MODEL), "model_sha256": DIGEST, **parameters}
    )


def native(pixels):
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel(str(MODEL))
    model.setModel("fsrcnn", 2)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model.upsample(pixels)


def source(encoding="gray", dtype="uint8"):
    yy, xx = np.indices((17, 23))
    base = ((xx * 41 + yy * 37) % 256).astype(np.uint8)
    if encoding != "gray":
        base = np.dstack((base, (base // 2), 255 - base))
        if encoding in ("bgra", "rgba"):
            base = np.dstack((base, np.tile(np.linspace(0, 255, 23).astype(np.uint8), (17, 1))))
    return ImageArray(convert_depth(base, dtype), encoding)


@pytest.mark.parametrize("encoding", ["gray", "bgr", "rgb", "bgra", "rgba"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_native_pixels_preserve_encoding_depth_and_interpolated_alpha(encoding, dtype):
    image = source(encoding, dtype)
    before = image.copy()
    pixels = convert_depth(opencv_image(image), "uint8")
    if encoding in ("bgra", "rgba"):
        pixels = pixels[..., :3]
    expected = native(np.ascontiguousarray(pixels))
    if encoding in ("rgb", "rgba"):
        expected = expected[..., ::-1]
    expected = convert_depth(expected, dtype)
    block = tool(depth_mode="convert_full_scale")
    try:
        result = block.execute({"image": image})
        assert result is not None, block.error
        output = result.outputs["image"]
        assert output.color_format == encoding and output.dtype == image.dtype
        baseline = cv2.resize(np.asarray(image), (46, 34), interpolation=cv2.INTER_CUBIC)
        if dtype == "float32":
            baseline = np.clip(baseline, 0, 1)
        np.testing.assert_array_equal(result.outputs["baseline"], baseline)
        if encoding in ("bgra", "rgba"):
            np.testing.assert_array_equal(output[..., :3], expected)
            np.testing.assert_array_equal(output[..., 3], baseline[..., 3])
        else:
            np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(image, before)
        assert result.outputs["info"]["quantized"] == (dtype != "uint8")
        assert result.outputs["info"]["alpha_inferred"] is False
    finally:
        block.close()


def test_nominal_grid_ticket_recipe_pyramid_and_auxiliary_payload_are_retained():
    image = source()
    context = {
        "spatial": frame([23, 17], "camera"),
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "part", "revision": 2, "digest": "a" * 64},
        "part": {"index": 12},
        "pyramid": {
            "level": 1,
            "largest_size": [46, 34],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
    }
    block = tool()
    try:
        result = block.execute({"image": ValuePacket(image, context)})
        assert result is not None, block.error
        for name, payload in result.payloads.items():
            assert payload["ticket"] == context["ticket"], name
            assert payload["recipe_ref"] == context["recipe_ref"]
            assert payload["part"] == context["part"]
            assert payload["spatial"]["image_size"] == [46, 34]
            assert payload["spatial"]["root_size"] == [23, 17]
            matrix = np.asarray(payload["spatial"]["to_root"])
            np.testing.assert_allclose(matrix @ [0, 0, 1], [-0.25, -0.25, 1])
            np.testing.assert_allclose(matrix @ [45, 33, 1], [22.25, 16.25, 1])
            np.testing.assert_allclose(payload["pyramid"]["to_largest"], np.eye(3))
    finally:
        block.close()


def test_mask_and_region_intersect_without_removing_context_or_quantizing_outside():
    image = source(dtype="uint16")
    mask = np.zeros(image.shape, np.uint8)
    mask[2:13, 4:20] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [23, 17],
        "shape": "rectangle",
        "rectangle": [8, 1, 14, 7],
    }
    block = tool(depth_mode="convert_full_scale")
    try:
        result = block.execute({"image": image, "mask": mask, "region": region})
        assert result is not None, block.error
        selected = np.zeros(image.shape, bool)
        selected[2:8, 8:20] = True
        selected = selected.repeat(2, 0).repeat(2, 1)
        baseline = cv2.resize(np.asarray(image), (46, 34), interpolation=cv2.INTER_CUBIC)
        full = native(convert_depth(image, "uint8")).astype(np.uint16) * 257
        expected = baseline.copy()
        expected[selected] = full[selected]
        np.testing.assert_array_equal(result.outputs["image"], expected)
        np.testing.assert_array_equal(
            result.outputs["processed_mask"], selected.astype(np.uint8) * 255
        )
        assert result.outputs["info"]["selected_source_pixels"] == 72
    finally:
        block.close()


def test_empty_selection_reads_no_model_and_returns_baseline():
    block = SuperResolution()
    result = block.execute({"image": source(), "mask": np.zeros((17, 23), np.uint8)})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], result.outputs["baseline"])
    assert result.outputs["info"]["native_calls"] == 0
    assert result.outputs["info"]["artifact"] is None
    block.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"scale": 8},
        {"scale": 2.0},
        {"maximum_input_pixels": 0},
        {"maximum_output_pixels": True},
        {"model_sha256": "ABC"},
        {"model_path": "../outside.pb"},
        {"algorithm": "unknown"},
        {"backend": "cuda", "target": "cpu"},
        {"backend": "opencv", "target": "cuda"},
        {"depth_mode": "silent"},
        {"baseline_interpolation": "bad"},
    ],
)
def test_parameter_errors_are_contained_and_recover(changes):
    block = tool(**changes)
    assert block.execute({"image": source()}) is None
    assert block.error
    block.parameters.update(copy.deepcopy(tool().parameters))
    assert block.execute({"image": source()}) is not None, block.error
    block.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_input_pixels": 100},
        {"maximum_output_pixels": 400},
        {"maximum_model_bytes": 100},
        {"model_sha256": "0" * 64},
        {"model_sha256": ""},
    ],
)
def test_budgets_and_model_identity_fail_before_native_creation(changes, monkeypatch):
    monkeypatch.setattr(
        cv2.dnn_superres,
        "DnnSuperResImpl_create",
        lambda: pytest.fail("native creation must not occur"),
    )
    block = tool(**changes)
    assert block.execute({"image": source()}) is None
    assert block.error
    block.close()


@pytest.mark.parametrize(
    "image",
    [
        source(dtype="uint16"),
        ImageArray(np.full((5, 8), 1.2, np.float32), "gray"),
        ImageArray(np.full((5, 8), np.nan, np.float32), "gray"),
        ImageArray(np.zeros((5, 8, 3), np.uint8), "hsv"),
        ImageArray(np.zeros((5, 8, 3), np.uint8), "gray"),
        ImageArray(np.zeros((5, 8), np.uint8), "rgb"),
        DepthArray(np.ones((5, 8))),
    ],
)
def test_unsupported_depth_range_encoding_and_physical_depth_reject(image):
    block = tool()
    assert block.execute({"image": image}) is None
    assert block.error
    block.close()


def test_pinned_snapshot_cache_tampering_cleanup_and_recovery(tmp_path, monkeypatch):
    raw = MODEL.read_bytes()
    path = tmp_path / "weights.pb"
    path.write_bytes(raw)
    real_create = cv2.dnn_superres.DnnSuperResImpl_create
    snapshots = []
    mutate = [True]

    class Observed:
        def __init__(self):
            self.native = real_create()

        def __getattr__(self, name):
            return getattr(self.native, name)

        def readModel(self, filename):
            snapshot = Path(filename)
            assert snapshot != path and snapshot.read_bytes() == raw
            snapshots.append(snapshot)
            if mutate[0]:
                path.write_bytes(b"x" * len(raw))
                mutate[0] = False
            self.native.readModel(filename)

    monkeypatch.setattr(cv2.dnn_superres, "DnnSuperResImpl_create", Observed)
    block = tool(model_path=str(path))
    try:
        first = block.execute({"image": source()})
        assert first is not None, block.error
        assert all(not p.exists() for p in snapshots)
        assert block.execute({"image": source()}) is None
        assert block._model is None and len(snapshots) == 1
        path.write_bytes(raw)
        assert block.execute({"image": source()}) is not None
        again = block.execute({"image": source()})
        assert again.outputs["info"]["artifact"]["native_model_reused"]
        assert len(snapshots) == 2
        block.parameters["target"] = "opencl"
        monkeypatch.setattr(cv2.dnn, "getAvailableTargets", lambda *a: [])
        assert block.execute({"image": source()}) is None
        assert "unavailable" in block.error and block._model is None
    finally:
        block.close()
        block.close()
    assert all(not p.parent.exists() for p in snapshots)


def test_scale_mismatch_clears_handle_and_recovers_with_correct_scale():
    block = tool(scale=3)
    assert block.execute({"image": source()}) is None
    assert block.error and block._model is None
    block.parameters["scale"] = 2
    assert block.execute({"image": source()}) is not None, block.error
    block.close()


def test_independent_models_execute_concurrently():
    def run(index):
        block = tool()
        try:
            result = block.execute({"image": source() + index})
            assert result is not None, block.error
            return result.outputs["image"]
        finally:
            block.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, range(4)))
    assert len(outputs) == 4 and not np.array_equal(outputs[0], outputs[-1])


def test_native_import_exception_cleans_snapshot_and_recovers(tmp_path, monkeypatch):
    from opencv_dataflow import super_resolution

    real_temporary = super_resolution.tempfile.TemporaryDirectory
    directories = []

    def temporary(**kwargs):
        directory = real_temporary(dir=tmp_path, **kwargs)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(super_resolution.tempfile, "TemporaryDirectory", temporary)
    path = tmp_path / "malformed.pb"
    raw = b"not a frozen TensorFlow graph"
    path.write_bytes(raw)
    block = tool(model_path=str(path), model_sha256=hashlib.sha256(raw).hexdigest())
    try:
        assert block.execute({"image": source()}) is None
        assert block.error and block._model is None
        assert len(directories) == 1 and not directories[0].exists()
        block.parameters.update(model_path=str(MODEL), model_sha256=DIGEST)
        assert block.execute({"image": source()}) is not None, block.error
    finally:
        block.close()
    assert len(directories) == 2 and not any(path.exists() for path in directories)


def test_selection_frame_mismatch_is_contained():
    block = tool()
    args = {
        "image": ValuePacket(source(), {"spatial": frame([23, 17], "camera")}),
        "mask": ValuePacket(
            np.ones((17, 23), np.uint8), {"spatial": frame([23, 17], "other-camera")}
        ),
    }
    try:
        assert block.execute(args) is None
        assert "frame differs" in block.error
    finally:
        block.close()


def test_cpu_does_not_initialize_other_device_contexts(monkeypatch):
    monkeypatch.setattr(
        cv2.dnn,
        "getAvailableTargets",
        lambda *args: pytest.fail("CPU must not enumerate unused devices"),
    )
    block = tool()
    try:
        assert block.execute({"image": source()}) is not None, block.error
    finally:
        block.close()
