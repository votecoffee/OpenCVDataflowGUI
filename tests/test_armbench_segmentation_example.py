import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples/Simple Plugins/pixel_segmentation/armbench_tote"
)


def test_armbench_replays_loaded_and_empty_totes_after_relocation(tmp_path):
    pytest.importorskip("opencv_defect_segmentation.dataflow")
    root = shutil.copytree(EXAMPLE, tmp_path / "Moved ARMBench example")
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        for sample in validation:
            graph.run()
            segment = graph.blocks["segment"]
            color = graph.blocks["color_overlay"]
            assert segment.state == "pass", segment.error
            assert color.state == "pass", color.error
            result = segment.last_result.outputs
            labels = cv2.imdecode(
                np.frombuffer((root / sample["labels"]).read_bytes(), np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            expected = labels == 2
            predicted = result["mask"] != 0
            np.testing.assert_array_equal(result["inspected"] != 0, labels != 0)
            scores = result["score_map"]
            assert np.isfinite(scores).all()
            assert np.all(scores[labels == 0] == 0)
            original = graph.blocks["source"].last_result.outputs["image"]
            overlay = color.last_result.outputs["overlay"]
            if expected.any():
                intersection = np.count_nonzero(predicted & expected)
                union = np.count_nonzero(predicted | expected)
                assert intersection / union > .92
                assert not np.array_equal(overlay, original)
            else:
                assert not predicted.any()
                np.testing.assert_array_equal(overlay, original)
        assert graph.sources_exhausted()
    finally:
        graph.close()


def test_armbench_pick_reports_clearance_then_clears_candidate_on_empty_tote(tmp_path):
    pytest.importorskip("opencv_defect_segmentation.dataflow")
    root = shutil.copytree(EXAMPLE, tmp_path / "Moved ARMBench example")
    graph = Graph.load(root / "pick_candidate.json", builtin_registry())
    try:
        graph.run()
        pick = graph.blocks["pick"]
        composite = graph.blocks["pick_overlay"]
        assert pick.state == "pass", pick.error
        assert composite.state == "pass", composite.error
        result = pick.last_result.outputs
        x, y = result["maximum_location"]
        mask = graph.blocks["segment"].last_result.outputs["mask"]
        labels = cv2.imdecode(
            np.frombuffer((root / "assets/OAR1KXNIME-labels.png").read_bytes(), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        assert mask[y, x] != 0 and labels[y, x] == 2
        distance = graph.blocks["distance"].last_result.outputs["distance"]
        assert result["value"] == pytest.approx(float(distance.max()))
        assert result["value"] == pytest.approx(95.5, abs=1.0)
        assert result["selected_count"] > 0
        np.testing.assert_array_equal(composite.last_result.outputs["image"][y, x], [85, 85, 250])

        graph.run()
        assert pick.state == "pass", pick.error
        assert composite.state == "pass", composite.error
        result = pick.last_result.outputs
        assert result["selected_count"] == 0
        assert result["value"] is NO_VALUE
        assert result["maximum_location"] is NO_VALUE
        np.testing.assert_array_equal(
            composite.last_result.outputs["image"],
            graph.blocks["source"].last_result.outputs["image"],
        )
        assert graph.sources_exhausted()
    finally:
        graph.close()
