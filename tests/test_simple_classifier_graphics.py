from types import SimpleNamespace

import numpy as np
import pytest

from opencv_dataflow.simple_classifier.graphics import draw_assessment
from opencv_dataflow.simple_classifier.model import ranked_scores


def test_chart_keeps_input_visible_and_bounds_many_long_class_labels():
    source = np.full((60, 80, 3), (12, 83, 191), np.uint8)
    prediction = ranked_scores([f"Class {i} " + "x" * 120 for i in range(35)], list(range(1, 36)))
    chart = draw_assessment(source, prediction)
    assert chart.shape == (1110, 1120, 3)
    assert chart.dtype == np.uint8
    # The source thumbnail remains unobscured by text/bars.
    assert np.count_nonzero(np.all(chart == source[0, 0], axis=2)) == 460 * 345
    assert np.all(source == (12, 83, 191))
    assert len(prediction["probabilities"]) == 35


def test_classifier_preview_is_optional_and_has_its_own_spatial_grid(monkeypatch, tmp_path):
    pytest.importorskip("opencv_dataflow")
    import opencv_dataflow.simple_classifier.dataflow as adapter
    from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, metadata

    prediction = ranked_scores(["First", "Second"], [1, 1])
    monkeypatch.setattr(
        adapter, "load_classifier", lambda *args: SimpleNamespace(predict=lambda im: prediction)
    )
    block = adapter.ClassifyImage(parameters={"model": "saved/manifest.json"})
    block.project_dir = tmp_path
    # Deliberately the same dimensions as the result chart: size alone
    # cannot decide whether source calibration applies to an output image.
    source = np.full((480, 1120), 14000, np.uint16)
    payload = {"spatial": frame([1120, 480], "camera")}
    inputs = {"image": ValuePacket(source, payload)}
    result = block.execute(inputs)
    assert result is not None, block.error
    assert block.state == "pass"
    assert block.filmstrip_output == "overlay"
    assert result.event_data["filmstrip_priority"] == -1
    assert metadata(result.outputs["overlay"])["spatial"]["root_id"] != "camera"
    assert result.outputs["confusion"] == 100
    assert result.score == .5
    assert "confusion" in result.message
    assert all(result.outputs[key] == value for key, value in prediction.items())
    assert np.all(source == 14000)

    def forbidden(*args):
        raise AssertionError("Disabled result rendering must not run")

    monkeypatch.setattr(adapter, "draw_assessment", forbidden)
    block.parameters["render_overlay"] = False
    no_image = block.execute(inputs)
    assert no_image is not None, block.error
    assert no_image.outputs["overlay"] is NO_VALUE
    assert block.filmstrip_output is None
    assert no_image.event_data == {}
    assert no_image.score == result.score
    assert all(no_image.outputs[key] == result.outputs[key] for key in prediction)
    block.parameters["render_overlay"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        block.validate_parameters()
