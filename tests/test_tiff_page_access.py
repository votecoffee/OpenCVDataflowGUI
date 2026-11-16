"""Native random pages, LRU budgets, snapshot refresh, tickets and worker persistence."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import ticket_key
from opencv_dataflow.tiff_page_access import TiffPageAccess
from opencv_dataflow.toolbox import block_entry

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/tiff_page_access"
STACK = BASE / "assets/mixed-stack.tiff"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def block(**parameters):
    return TiffPageAccess(parameters={"path": str(STACK), **parameters})


def read(tool, index=None, **inputs):
    if index is not None:
        inputs["page_index"] = index
    result = tool.execute(inputs)
    assert result is not None, tool.error
    return result


def test_random_nonsequential_pages_match_raw_hashes_and_only_misses_decode(monkeypatch):
    original = cv2.imdecodemulti
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs["range"])
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecodemulti", observed)
    tool = block(cache_pages=2, cache_bytes=512 * 1024)
    roots, tickets, peak = {}, set(), 0
    try:
        for index in TRUTH["request_order"]:
            result = read(tool, index)
            image, info = result.outputs["image"], result.outputs["info"]
            expected = TRUTH["pages"][index]
            assert hashlib.sha256(image.tobytes()).hexdigest() == expected["bgr_sha256"]
            assert image.dtype.name == expected["dtype"] and list(image.shape) == expected["shape"]
            assert result.outputs["page_count"] == 257
            assert info["page_index"] == index
            spatial = result.payloads["image"]["spatial"]
            if index in roots:
                assert spatial == roots[index]
            roots[index] = spatial
            tickets.add(ticket_key(result.outputs["ticket"]))
            assert len(info["cache"]["pages"]) <= 2
            assert info["cache"]["bytes"] <= 512 * 1024
            peak = max(peak, info["cache"]["bytes"])
            # Returned pixels are private; mutation must not poison a later hit.
            image[...] = 0
        assert calls == [(i, i + 1) for i in (256, 3, 128, 256, 255, 3, 0, 128, 256)]
        assert len(tickets) == len(TRUTH["request_order"])
        assert info["cache"]["snapshot_reads"] == 1
        assert info["cache"]["hits"] == 2 and info["cache"]["evictions"] == 7
        assert peak <= 196608 < TRUTH["raw_stack_bytes"]
        assert all(not value.flags.writeable for value in tool._cache.values())
    finally:
        tool.close()
    assert tool._data is None and not tool._cache


@pytest.mark.parametrize("parameters", [{"cache_pages": 0}, {"cache_bytes": 0}, {"cache_bytes": 1}])
def test_cache_can_be_disabled_or_bypassed_without_changing_samples(parameters):
    tool = block(**parameters)
    for _ in range(2):
        result = read(tool, 3)
        assert not result.outputs["info"]["cache"]["stored"]
        assert (
            hashlib.sha256(result.outputs["image"].tobytes()).hexdigest()
            == TRUTH["pages"][3]["bgr_sha256"]
        )
    assert result.outputs["info"]["cache"]["misses"] == 2
    assert not tool._cache
    tool.close()


@pytest.mark.parametrize("index", range(4))
def test_each_storage_layout_retains_exact_depth_channels_and_hdr(index):
    tool = block()
    result = read(tool, index)
    image = result.outputs["image"]
    assert hashlib.sha256(image.tobytes()).hexdigest() == TRUTH["pages"][index]["bgr_sha256"]
    if index == 3:
        assert image.min() < 0 and image.max() > 1 and image.dtype == np.float64
    hit = read(tool, index)
    np.testing.assert_array_equal(hit.outputs["image"], image)
    assert hit.outputs["info"]["cache"]["hit"]
    tool.close()


def test_byte_budget_lru_eviction_and_tightening_are_enforced():
    tool = block(cache_pages=4, cache_bytes=100000)
    read(tool, 0)
    current = read(tool, 3)  # float64 gray is 98,304 bytes; cannot coexist with uint16 page 0.
    assert current.outputs["info"]["cache"]["pages"] == [3]
    assert current.outputs["info"]["cache"]["bytes"] == 98304
    tool.parameters["cache_bytes"] = 24576
    current = read(tool, 0)
    assert current.outputs["info"]["cache"]["pages"] == [0]
    tool.parameters["cache_pages"] = 0
    assert not read(tool, 0).outputs["info"]["cache"]["stored"]
    assert tool._cached_bytes == 0
    tool.close()


@pytest.mark.parametrize("policy", ["on_change", "verify_content", "pinned"])
def test_snapshot_change_invalidation_pinning_and_reset(policy, tmp_path):
    path = tmp_path / "pages.tiff"

    def write(value):
        ok, data = cv2.imencodemulti(
            ".tiff", [np.full((9, 13), value, np.uint16)], [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        )
        assert ok
        path.write_bytes(data.tobytes())

    write(1000)
    tool = block(path=str(path), refresh=policy)
    first = read(tool)
    before = path.stat()
    write(2000)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
    second = read(tool)
    assert np.all(second.outputs["image"] == (1000 if policy == "pinned" else 2000))
    first_root = first.payloads["image"]["spatial"]["root_id"]
    assert (first_root == second.payloads["image"]["spatial"]["root_id"]) is (policy == "pinned")
    refreshed = read(tool, reset=True)
    assert np.all(refreshed.outputs["image"] == 2000)
    assert not refreshed.outputs["info"]["cache"]["hit"]
    tool.close()


def test_content_verification_keeps_identical_cache_and_detects_hidden_rewrite(
    tmp_path, monkeypatch
):
    path = tmp_path / "same.tiff"
    shutil.copyfile(STACK, path)
    tool = block(path=str(path), refresh="verify_content")
    first = read(tool, 0)
    again = read(tool, 0)
    assert again.outputs["info"]["cache"]["hit"]
    assert again.outputs["info"]["cache"]["snapshot_reads"] == 2
    assert again.outputs["info"]["cache"]["snapshot_loads"] == 1
    # Hide all file stamp differences; SHA verification still replaces the snapshot.
    import opencv_dataflow.tiff_page_access as module

    saved = tool._stamp
    monkeypatch.setattr(module, "file_stamp", lambda stat: saved)
    ok, data = cv2.imencodemulti(".tiff", [np.full((9, 13), 3000, np.uint16)])
    assert ok
    path.write_bytes(data.tobytes())
    changed = read(tool, 0)
    assert np.all(changed.outputs["image"] == 3000)
    assert changed.outputs["info"]["container_sha256"] != first.outputs["info"]["container_sha256"]
    assert changed.outputs["info"]["cache"]["snapshot_loads"] == 2
    tool.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"page_index": 257},
        {"maximum_page_pixels": 1},
        {"maximum_decoded_bytes": 1},
        {"maximum_encoded_bytes": 8},
        {"pixel_depth": "uint16", "color_format": "hsv"},
        {"page_index": 3, "color_format": "hsv"},
    ],
)
def test_invalid_request_is_rejected_before_native_pixel_decode(monkeypatch, parameters):
    calls = []
    monkeypatch.setattr(cv2, "imdecodemulti", lambda *args, **kwargs: calls.append(1))
    tool = block(**parameters)
    assert tool.execute({}) is None and tool.error_kind == "invalid"
    assert not calls and not tool.current_emissions
    tool.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"path": ""},
        {"page_index": -1},
        {"page_index": 1024},
        {"cache_bytes": -1},
        {"cache_pages": 65},
        {"cache_pages": 2.0},
        {"refresh": "guess"},
        {"page_ticket": "true"},
    ],
)
def test_static_controls_have_validation(parameters):
    tool = block(**parameters)
    assert tool.execute({}) is None and tool.error_kind == "invalid"
    assert not tool.current_emissions and tool._data is None


def test_page_format_changes_use_private_raw_cache_and_external_ticket_configuration():
    tool = block(page_index=1)
    first = read(tool)
    raw = first.outputs["image"].copy()
    tool.parameters.update(color_format="rgb", pixel_depth="uint16")
    converted = read(tool)
    np.testing.assert_array_equal(
        converted.outputs["image"], raw[..., ::-1].astype(np.uint16) * 257
    )
    assert converted.outputs["info"]["cache"]["hit"]
    assert converted.payloads["image"]["spatial"] == first.payloads["image"]["spatial"]
    tool.ticket_settings = {"mode": "external", "value": 1}
    ticketed = read(tool, ticket="plc-part-19")
    assert ticketed.outputs["ticket"]["value"] == "plc-part-19"
    tool.close()


def test_worker_cache_handoff_save_as_xml_and_editor(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    graph = Graph()
    registry = builtin_registry()
    tool = graph.add(
        registry.create(
            "input.tiff_page",
            block_id="page",
            parameters={
                "path": "assets/mixed-stack.tiff",
                "path_base": "project",
                "page_index": 256,
            },
        )
    )
    graph.bind_project_dir(BASE)
    graph.save(tmp_path / "saved.json")
    roots = []
    try:
        for run in range(3):
            worker = Graph.from_dict(graph.to_dict(), registry, project_dir=graph.project_dir)
            worker.copy_runtime_from(graph)
            worker.run()
            active = worker.blocks["page"]
            assert active.last_result is not None, active.error
            roots.append(active.last_result.payloads["image"]["spatial"])
            assert active.last_result.outputs["info"]["cache"]["hit"] is (run > 0)
            graph.copy_runtime_from(worker)
            tool.copy_results_from(active)
            worker.close()
            assert tool._data is not None and tool._cache
        assert roots[0] == roots[1] == roots[2]
        xml = tmp_path / "page.input.tiff_page.xml"
        save_block_xml(tool, xml)
        assert read_block_xml(xml)["parameters"] == tool.parameters
        reopened = Graph.load(tmp_path / "saved.json", registry)
        assert reopened.blocks["page"]._data is None
        reopened.close()
        assert block_entry(tool).path[:2] == ("OpenCV", "Image Codecs (imgcodecs)")
        window = MainWindow()
        try:
            window.replace_graph(graph)
            app.processEvents()
            window.inspector.select(tool)
            for name in tool.parameter_choices:
                assert isinstance(window.inspector.fields[name], QComboBox)
        finally:
            window.dirty = False
            window.close()
            app.processEvents()
    finally:
        graph.close()


def test_pinned_snapshot_survives_file_removal_until_reset_and_bad_request_can_recover(tmp_path):
    path = tmp_path / "páginas.tiff"
    shutil.copyfile(STACK, path)
    tool = block(path=str(path), refresh="pinned")
    first = read(tool, 0)
    path.unlink()
    assert read(tool, 0).outputs["info"]["cache"]["hit"]
    assert tool.execute({"page_index": 999}) is None and not tool.current_emissions
    assert (
        read(tool, 0).outputs["info"]["container_sha256"]
        == first.outputs["info"]["container_sha256"]
    )
    assert tool.execute({"reset": True}) is None and not tool.current_emissions
    assert tool._data is None and not tool._cache
    tool.close()


def test_native_failure_and_cancellation_do_not_publish_partial_images(monkeypatch):
    def broken(*args, **kwargs):
        raise cv2.error("Controlled page decoder failure")

    monkeypatch.setattr(cv2, "imdecodemulti", broken)
    tool = block()
    assert tool.execute({}) is None and not tool.current_emissions and not tool._cache
    assert tool.execute({}, cancelled=lambda: True) is None and not tool.current_emissions
    tool.close()
