"""Original animation projects execute through the normal saved-graph routes."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.runner import main
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/animation_codecs"


@pytest.mark.local_examples
def test_animation_batch_source_uses_the_normal_single_run_cli_route(tmp_path):
    report = tmp_path / "cli.json"
    assert main([str(EXAMPLE / "apng_roundtrip.json"), "--report", str(report)]) == 0
    assert report.is_file()


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "name",
    [
        "apng_roundtrip",
        "webp_roundtrip",
        "gif_timing",
        "gif_timing_failure",
        "frame_inspection",
        "write_webp",
    ],
)
def test_saved_animation_examples(name, profile, tmp_path):
    registry = builtin_registry()
    graph = Graph.load(EXAMPLE / (name + ".json"), registry)
    try:
        graph.set_run_profile(profile)
        if name == "write_webp":
            graph.blocks["write"].parameters["path"] = str(tmp_path / "new.webp")
        graph.run()
        active = list(graph.processing_blocks())
        if name == "gif_timing_failure":
            assert graph.blocks["encode"].state == "error"
            assert not graph.blocks["encode"].current_emissions
            assert graph.blocks["decode"].state in ("inactive", "skipped")
        else:
            assert all(b.state == "pass" for b in active), [(b.id, b.error) for b in active]
            if "verify" in graph.blocks:
                assert graph.blocks["verify"].last_result.outputs["value"] == 0
            if name == "frame_inspection":
                inspection = graph.blocks["inspect"]
                assert inspection.last_result.outputs["count"] == 1
                assert len(inspection.current_emissions["count"]) == 3
                np.testing.assert_allclose(inspection.last_result.outputs["centroids"], [[116, 50]])
            if name == "gif_timing":
                assert graph.blocks["decode"].last_result.outputs["durations"] == [80, 120, 200]
        # Normal Save As must retain access to the frozen animation input.
        graph.save(tmp_path / "saved.json")
        reopened = Graph.load(tmp_path / "saved.json", registry)
        try:
            source = reopened.blocks["source"]
            result = source.execute({})
            assert result is not None, source.error
            assert result.outputs["durations"] == [80, 120, 200]
        finally:
            reopened.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_independent_fixture_formula_and_design_truth():
    truth = json.loads((EXAMPLE / "truth.json").read_bytes())
    y, x = np.indices((96, 160))
    for index, center in enumerate(truth["circle_centers"]):
        data = (EXAMPLE / f"assets/frame_{index}.png").read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        expected = np.dstack((18 + x // 5 + index * 7, 25 + y // 4, 30 + (x + y) // 7)).astype(
            np.uint8
        )
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= truth["circle_radius_pixels"] ** 2
        expected[mask] = 230
        np.testing.assert_array_equal(image, expected)


def test_animation_toolbox_enums_and_inspector_draft_validation(app):
    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.parameter_validation import ValidationError

    registry = builtin_registry()
    block = registry.create("cv.encode_animation")
    entry = block_entry(type(block))
    assert entry.path == ("OpenCV", "Image Codecs (imgcodecs)", "Codecs")
    window = MainWindow()
    try:
        graph = Graph()
        graph.add(block)
        window.replace_graph(graph, None)
        window.inspector.select(block)
        for key in (
            "codec",
            "gif_quality",
            "gif_dither",
            "gif_color_table",
            "webp_mode",
            "png_strategy",
        ):
            assert isinstance(window.inspector.fields[key], QComboBox)
        invalid = dict(block.parameters, background_bgra=[0, 0, 300, 0])
        with pytest.raises(ValidationError) as error:
            block.validate_parameters(invalid)
        assert error.value.fields == ("background_bgra",)
    finally:
        window.dirty = False
        window.close()
