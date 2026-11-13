"""Known geometry and material-band truth in the public, original fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

pytest.importorskip("skimage")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "custom tools"


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("name", ["skimage_region_properties", "skimage_thresholding"])
def test_public_example_truth_and_profiles(name, profile):
    folder = EXAMPLES / name
    expected = json.loads((folder / "expected.json").read_text(encoding="utf8"))
    graph = Graph.load(folder / (name + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks()), [
            (b.id, b.state, b.error) for b in graph.processing_blocks()
        ]
        result = graph.blocks["measure"].last_result
        if name == "skimage_region_properties":
            items = result.outputs["collection"]["items"]
            assert len(items) == 3
            for item, truth in zip(items, expected["regions"], strict=True):
                metrics = item["metrics"]
                assert item["id"] == truth["label"]
                assert metrics["area"] == truth["area_pixels"]
                np.testing.assert_allclose(
                    [metrics["center_x"], metrics["center_y"]], truth["centroid_xy"]
                )
                assert metrics["regionprops.area"] == pytest.approx(truth["area_pixels"] * 0.01)
                assert metrics["regionprops.euler_number"] == truth["euler_number"]
            ring = graph.blocks["ring_mask"].last_result.outputs["mask"]
            np.testing.assert_equal(
                ring, (result.outputs["labels"] == expected["retained_label"]) * 255
            )
            if profile == "lean":
                assert not result.drawings
            else:
                assert result.drawings
        else:
            multi = graph.blocks["multi"].last_result.outputs
            labels = multi["labels"]
            truth = np.repeat(np.arange(3), 60)[None, :]
            accuracy = float((labels == truth).mean())
            assert accuracy >= expected["minimum_classification_accuracy"]
            assert sum(multi["statistics"]["class_counts"]) == labels.size
            assert len(result.outputs["collection"]["items"]) == 3
            assert graph.blocks["source"].last_result.outputs["image"].dtype == np.uint16
            for kind in ("gaussian", "niblack", "sauvola"):
                local = graph.blocks["local_" + kind].last_result.outputs
                assert local["threshold_map"].dtype == np.float64
                assert local["threshold_map"].max() > 255
    finally:
        graph.close()
