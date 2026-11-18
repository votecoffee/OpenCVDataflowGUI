"""Native scale pixels, one-forward delivery, provenance, failure bounds and saved graphs."""

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, convert_depth, opencv_image
from opencv_dataflow.multiscale_super_resolution import MultiScaleSuperResolution
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/multiscale_super_resolution"
MODEL = BASE / "assets/LapSRN_x8.pb"
DIGEST = "a00211938b2b4c7d23691525a9267d9d0bbb345cdb9f38a7fc917414a5a1884f"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def block(**params):
    return MultiScaleSuperResolution(
        parameters={"model_path": str(MODEL), "model_sha256": DIGEST, **params}
    )


def source(encoding="gray", dtype="uint8"):
    yy, xx = np.indices((11, 13))
    image = ((xx * 41 + yy * 17) % 256).astype(np.uint8)
    if encoding != "gray":
        image = np.dstack((image, image // 2, 255 - image))
        if encoding in ("bgra", "rgba"):
            image = np.dstack((image, ((xx * 7 + yy * 15) % 256).astype(np.uint8)))
    return ImageArray(convert_depth(image, dtype), encoding)


def native(pixels, scales):
    cv2.dnn_superres.DnnSuperResImpl_create()
    network = cv2.dnn.readNetFromTensorflow(str(MODEL), "", cv2.dnn.ENGINE_CLASSIC)
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    working = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb) if pixels.ndim == 3 else pixels
    working = cv2.addWeighted(working, 1 / 255, working, 0, 0, dtype=cv2.CV_32F)
    y = working[..., 0] if pixels.ndim == 3 else working
    network.setInput(cv2.dnn.blobFromImage(y))
    output = []
    for scale in scales:
        tensor = network.forward(f"NCHW_output_{scale}x")[0, 0]
        if pixels.ndim == 3:
            size = (pixels.shape[1] * scale, pixels.shape[0] * scale)
            tensor = cv2.merge(
                [tensor, cv2.resize(working[..., 1], size), cv2.resize(working[..., 2], size)]
            )
        converted = cv2.addWeighted(tensor, 255, tensor, 0, 0, dtype=cv2.CV_8U)
        output.append(
            cv2.cvtColor(converted, cv2.COLOR_YCrCb2BGR) if pixels.ndim == 3 else converted
        )
    return output


@pytest.mark.parametrize("encoding", ["gray", "bgr", "rgb", "bgra", "rgba"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_all_scale_pixels_match_independent_named_native_heads_and_8x_native_wrapper(
    encoding, dtype
):
    image = source(encoding, dtype)
    before = image.copy()
    pixels = convert_depth(opencv_image(image), "uint8")
    if encoding in ("bgra", "rgba"):
        pixels = pixels[..., :3]
    expected = native(np.ascontiguousarray(pixels), [2, 4, 8])
    b = block(depth_mode="convert_full_scale")
    try:
        r = b.execute({"image": image})
        assert r is not None, b.error
        assert r.outputs["info"]["native_calls"] == 1 and r.outputs["images"].labels == (
            "x2",
            "x4",
            "x8",
        )
        for s, value in zip((2, 4, 8), expected, strict=True):
            if encoding in ("rgb", "rgba"):
                value = value[..., ::-1]
            value = convert_depth(value, dtype)
            actual = r.outputs[f"image_x{s}"]
            assert actual.dtype == image.dtype and actual.color_format == encoding
            np.testing.assert_array_equal(
                actual[..., :3] if encoding in ("bgra", "rgba") else actual, value
            )
            if encoding in ("bgra", "rgba"):
                baseline = cv2.resize(
                    np.asarray(image), (13 * s, 11 * s), interpolation=cv2.INTER_CUBIC
                )
                if dtype == "float32":
                    baseline = np.clip(baseline, 0, 1)
                np.testing.assert_array_equal(actual[..., 3], baseline[..., 3])
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(MODEL))
        sr.setModel("lapsrn", 8)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        np.testing.assert_array_equal(sr.upsample(np.ascontiguousarray(pixels)), expected[-1])
        np.testing.assert_array_equal(image, before)
    finally:
        b.close()


def test_group_and_ports_retain_individual_grids_pyramid_recipe_ticket_and_auxiliary_data():
    image = source()
    ctx = {
        "spatial": frame([13, 11], "camera"),
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "part", "revision": 2, "digest": "a" * 64},
        "custom": {"part": 14},
        "pyramid": {
            "level": 1,
            "largest_size": [26, 22],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
    }
    b = block(
        output_scales=[8, 2], output_nodes=["NCHW_output_8x", "NCHW_output_2x"], preview_scale="2"
    )
    try:
        r = b.execute({"image": ValuePacket(image, ctx)})
        assert r is not None, b.error
        assert r.outputs["images"].labels == ("x8", "x2") and r.outputs["image"].shape == (22, 26)
        assert r.outputs["image_x4"] is NO_VALUE
        for packet, s in zip(r.outputs["images"].items, [8, 2], strict=True):
            p = packet.payload
            matrix = np.array([[1 / s, 0, (1 / s - 1) / 2], [0, 1 / s, (1 / s - 1) / 2], [0, 0, 1]])
            assert p["spatial"]["image_size"] == [13 * s, 11 * s] and p["spatial"]["root_size"] == [
                13,
                11,
            ]
            np.testing.assert_allclose(p["spatial"]["to_root"], matrix)
            np.testing.assert_allclose(
                p["pyramid"]["to_largest"], np.array(ctx["pyramid"]["to_largest"]) @ matrix
            )
            assert (
                p["ticket"] == ctx["ticket"]
                and p["recipe_ref"] == ctx["recipe_ref"]
                and p["custom"] == ctx["custom"]
            )
            assert p["super_resolution"]["scale"] == s
            assert r.payloads[f"image_x{s}"]["spatial"] == p["spatial"]
        assert (
            r.payloads["info"]["spatial"] == ctx["spatial"]
            and r.payloads["images"]["spatial"] == ctx["spatial"]
        )
    finally:
        b.close()


def test_one_forward_exact_order_cache_rebinding_and_no_second_fallback(monkeypatch):
    import opencv_dataflow.multiscale_super_resolution as module

    original = module.cv2.dnn.readNetFromTensorflow
    calls = {"load": 0, "forward": 0}

    class Proxy:
        def __init__(self, net):
            self.net = net

        def __getattr__(self, name):
            return getattr(self.net, name)

        def forward(self, names):
            calls["forward"] += 1
            assert names == ["NCHW_output_8x", "NCHW_output_2x"]
            return self.net.forward(names)

    def load(*args, **kwargs):
        calls["load"] += 1
        return Proxy(original(*args, **kwargs))

    monkeypatch.setattr(module.cv2.dnn, "readNetFromTensorflow", load)
    b = block(output_scales=[8, 2], output_nodes=["NCHW_output_8x", "NCHW_output_2x"])
    try:
        a = b.execute({"image": source()})
        assert a is not None, b.error
        handle = b._model
        c = b.execute({"image": ImageArray(255 - source(), "gray")})
        assert c is not None, b.error
        assert calls == {"load": 1, "forward": 2} and b._model is handle
        assert c.outputs["info"]["artifact"]["native_model_reused"]
        assert not np.array_equal(a.outputs["image_x2"], c.outputs["image_x2"])
    finally:
        b.close()


def test_empty_selection_does_not_read_model_and_every_scale_is_baseline():
    image = source()
    b = block(model_path="missing.pb", model_sha256="a" * 64)
    r = b.execute({"image": image, "mask": np.zeros(image.shape, np.uint8)})
    assert r is not None, b.error
    assert (
        r.outputs["info"]["native_calls"] == 0
        and r.outputs["info"]["artifact"] is None
        and b._model is None
    )
    for s in (2, 4, 8):
        np.testing.assert_array_equal(
            r.outputs[f"image_x{s}"],
            cv2.resize(np.asarray(image), (13 * s, 11 * s), interpolation=cv2.INTER_CUBIC),
        )


def test_later_native_head_failure_is_atomic_and_clears_cached_network():
    b = block()
    try:
        previous = b.execute({"image": source()})
        assert previous is not None, b.error
        original = b._model

        class BrokenLastHead:
            def __getattr__(self, name):
                return getattr(original, name)

            def forward(self, names):
                values = list(original.forward(names))
                values[-1] = values[-1].copy()
                values[-1][0, 0, 0, 0] = np.nan
                return values

        b._model = BrokenLastHead()
        assert b.execute({"image": source()}) is None and not b.current_emissions
        assert b._model is None and b.last_result is None
    finally:
        b.close()


def test_region_and_mask_intersect_full_context_and_baseline_elsewhere():
    image = source()
    mask = np.zeros(image.shape, np.uint8)
    mask[:, :8] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [13, 11],
        "shape": "rectangle",
        "rectangle": [3, 0, 7, 11],
        "invert": False,
    }
    full = block()
    selected = block()
    try:
        expected = full.execute({"image": image})
        r = selected.execute({"image": image, "mask": mask, "region": region})
        assert r is not None, selected.error
        assert r.outputs["info"]["selected_source_pixels"] == 5 * 11
        for s in (2, 4, 8):
            a = r.outputs[f"image_x{s}"]
            baseline = cv2.resize(
                np.asarray(image), (13 * s, 11 * s), interpolation=cv2.INTER_CUBIC
            )
            np.testing.assert_array_equal(a[:, : 3 * s], baseline[:, : 3 * s])
            np.testing.assert_array_equal(a[:, 8 * s :], baseline[:, 8 * s :])
            np.testing.assert_array_equal(
                a[:, 3 * s : 8 * s], expected.outputs[f"image_x{s}"][:, 3 * s : 8 * s]
            )
    finally:
        full.close()
        selected.close()


@pytest.mark.parametrize(
    "params",
    [
        {"output_scales": [2]},
        {"output_scales": [2, 2]},
        {"output_scales": [True, 4]},
        {"output_nodes": ["a", "a", "b"]},
        {"output_nodes": ["a", "b"]},
        {"preview_scale": "4", "output_scales": [2, 8], "output_nodes": ["a", "b"]},
        {"maximum_output_pixels": True},
        {"model_sha256": "A" * 64},
        {"model_path": "../model.pb"},
    ],
)
def test_invalid_settings_validate_without_native_work(params):
    with pytest.raises(ValidationError):
        block(**params).validate_parameters()


@pytest.mark.parametrize(
    "params",
    [
        {"maximum_input_pixels": 1},
        {"maximum_output_pixels": 1},
        {"maximum_model_bytes": 1},
        {"model_sha256": "a" * 64},
        {"output_nodes": ["missing", "NCHW_output_4x", "NCHW_output_8x"]},
        {"output_nodes": ["NCHW_output_4x", "NCHW_output_2x", "NCHW_output_8x"]},
    ],
)
def test_invalid_runtime_models_shapes_and_budgets_emit_no_partial_group(params):
    b = block(**params)
    try:
        assert b.execute({"image": source()}) is None and not b.current_emissions and b.error
        assert b._model is None
    finally:
        b.close()


def test_native_import_cleanup_error_recovery_and_failed_input_admission(tmp_path, monkeypatch):
    import opencv_dataflow.multiscale_super_resolution as module

    actual = module.cv2.dnn.readNetFromTensorflow
    paths = []

    def failed(path, *args):
        paths.append(Path(path))
        assert Path(path).read_bytes() == MODEL.read_bytes()
        raise cv2.error("controlled import failure")

    monkeypatch.setattr(module.cv2.dnn, "readNetFromTensorflow", failed)
    b = block()
    try:
        assert b.execute({"image": source()}) is None and b._model is None
        assert paths and all(not p.exists() for p in paths)
        monkeypatch.setattr(module.cv2.dnn, "readNetFromTensorflow", actual)
        r = b.execute({"image": source()})
        assert r is not None, b.error
        previous = b.last_inputs
        handle = b._model
        b.input_failure_policies["image"] = "reject_failed"
        b.execute(
            {
                "image": ValuePacket(
                    source(),
                    {
                        "outcome": {
                            "model": "tool_outcome_v1",
                            "passed": False,
                            "chain_passed": False,
                            "source": {"id": "prior"},
                            "fail_reasons": [],
                        }
                    },
                )
            }
        )
        assert (
            b.input_rejected
            and not b.current_emissions
            and b.last_inputs is previous
            and b.last_result is r
            and b._model is handle
        )
    finally:
        b.close()


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("mode", ["debug", "lean"])
def test_saved_projects_scales_root_localization_and_filmstrip(case, mode):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(mode)
        graph.run()
        b = graph.blocks["learned"]
        r = b.last_result
        if case in ("wrong_scale", "budget_error"):
            assert r is None and b.error and not b.current_emissions
            return
        assert r is not None, b.error
        assert r.outputs["info"]["native_calls"] == (0 if case == "empty_region" else 1)
        assert (
            r.outputs["info"]["artifact"] is None
            if case == "empty_region"
            else r.outputs["info"]["artifact"]["sha256"] == DIGEST
        )
        group = r.outputs["images"]
        assert group.labels == tuple(f"x{s}" for s in b.parameters["output_scales"])
        for packet, scale in zip(group.items, b.parameters["output_scales"], strict=True):
            assert packet.value.shape[:2] == (24 * scale, 32 * scale)
            assert ticket_key(packet.payload["ticket"]) == ticket_key(
                graph.blocks["source"].last_result.outputs["ticket"]
            )
        assert (
            bool(r.drawings) == (mode == "debug")
            and graph.blocks["filmstrip"].last_result is not None
        )
        if case == "geometry":
            for s in (2, 4, 8):
                found = max(
                    graph.blocks[f"blobs_{s}"].last_result.outputs["collection"]["items"],
                    key=lambda o: o["metrics"]["area"],
                )
                center = np.array([found["metrics"]["center_x"], found["metrics"]["center_y"], 1])
                payload = graph.blocks[f"blobs_{s}"].last_result.payloads["collection"]
                root = np.array(payload["spatial"]["to_root"]) @ center
                assert np.linalg.norm(root[:2] - [70, 88]) < 1
    finally:
        graph.close()


def test_model_assets_regeneration_and_separate_native_benchmark(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path)
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (BASE / p.relative_to(tmp_path)).read_bytes(), p
    with pytest.raises(FileExistsError):
        build(tmp_path)
    measured = runpy.run_path(str(BASE / "benchmark.py"))["benchmark"]()
    frozen = json.loads((BASE / "benchmark.json").read_bytes())
    for case in ("reference", "noise"):
        for scale in ("2", "4", "8"):
            for method in ("learned", "linear", "cubic"):
                for metric, value in measured["results"][case][scale][method].items():
                    np.testing.assert_allclose(
                        value, frozen["results"][case][scale][method][metric], atol=1e-7
                    )
    assert measured["results"]["reference"]["8"]["learned"]["erased_dark_detail_pixels"] > 200


def test_portable_json_xml_assets_license_and_no_persisted_cache(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "reordered.json", registry)
    try:
        graph.run()
        expected = graph.blocks["learned"].last_result.outputs["image"].copy()
        assert block_entry(graph.blocks["learned"]).path[:2] == (
            "OpenCV",
            "Deep Neural Networks (dnn)",
        )
        graph.save(tmp_path / "saved.json")
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
        assert (tmp_path / "assets/LapSRN_x8.pb").read_bytes() == MODEL.read_bytes()
        assert (tmp_path / "MODEL-LICENSE.txt").read_bytes() == (
            BASE / "MODEL-LICENSE.txt"
        ).read_bytes()
        assert (tmp_path / "SOURCES.md").read_bytes() == (BASE / "SOURCES.md").read_bytes()
    finally:
        graph.close()
    for graph in (
        Graph.load(tmp_path / "saved.json", registry),
        Graph.from_dict(read_project_xml(tmp_path / "saved.xml"), registry, project_dir=tmp_path),
    ):
        try:
            assert graph.blocks["learned"]._model is None
            graph.run()
            np.testing.assert_array_equal(
                graph.blocks["learned"].last_result.outputs["image"], expected
            )
        finally:
            graph.close()


def test_cold_offscreen_worker_releases_native_graph():
    script = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry();graph=Graph.load(Path(sys.argv[1]),registry)
try:
 worker=Runner(graph,'once',registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(20000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==1
 assert graph.blocks['learned'].last_result.outputs['info']['native_calls']==1
finally:graph.close()
"""
    p = subprocess.run(
        [sys.executable, "-c", script, str(BASE / "geometry.json")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=35,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert p.returncode == 0, p.stdout + p.stderr


def test_pinned_native_multioutput_binding_has_no_python_output_delivery():
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODEL))
    sr.setModel("lapsrn", 8)
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    outputs = []
    returned = sr.upsampleMultioutput(
        np.asarray(source()),
        outputs,
        [2, 4, 8],
        ["NCHW_output_2x", "NCHW_output_4x", "NCHW_output_8x"],
    )
    assert returned is None and outputs == []
