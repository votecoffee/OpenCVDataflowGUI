import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import DepthArray, ImageArray, preview_rgb
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolblocks import encapsulate


@pytest.mark.parametrize(
    "encoding,labels", [("bgr", ["B", "G", "R"]), ("rgb", ["R", "G", "B"]), ("gray", ["Gray"])]
)
def test_histogram_channel_counts(encoding, labels):
    pixels = np.array([[0, 0, 255], [255, 128, 128]], np.uint8)
    image = pixels if encoding == "gray" else np.stack((pixels, pixels // 2, pixels // 3), axis=2)
    block = builtin_registry().create("inspect.histogram")
    block.execute({"image": ImageArray(image, encoding)})
    assert block.state == "pass", block.error
    result = block.last_result.outputs["histogram"]
    assert [c["channel"] for c in result["channels"]] == labels
    for index, channel in enumerate(result["channels"]):
        expected = pixels if encoding == "gray" else image[..., index]
        np.testing.assert_array_equal(
            channel["counts"], np.bincount(expected.ravel(), minlength=256)
        )
        assert sum(channel["counts"]) == 6
    assert block.last_output_images["plot"].shape == (300, 600, 3)


def test_histogram_float_nan_manual_range_and_alpha():
    block = builtin_registry().create(
        "inspect.histogram",
        parameters={"range_mode": "manual", "range_min": 0.0, "range_max": 1.0, "bins": 2},
    )
    block.execute({"image": np.array([[np.nan, -1, 0, 0.5, 1, 2]], np.float32)})
    assert block.state == "pass", block.error
    channel = block.last_result.outputs["histogram"]["channels"][0]
    assert channel["counts"] == [1, 2] and channel["outside_range"] == 2
    block.parameters.update(range_mode="automatic", include_alpha=True)
    block.execute({"image": np.zeros((2, 2, 4), np.uint16)})
    assert len(block.last_result.outputs["histogram"]["channels"]) == 4


def test_percentiles_and_summary_use_selected_samples_independent_of_bin_range():
    block = builtin_registry().create(
        "inspect.histogram",
        parameters={
            "bins": 2, "range_mode": "manual", "range_max": 5.0,
            "lower_percentile": 25, "upper_percentile": 75, "threshold_statistic": "mean",
        },
    )
    image = np.array([[0, 2, 10, 14, np.nan, 500]], np.float32)
    mask = np.array([[255, 255, 255, 255, 255, 0]], np.uint8)
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    channel = result.outputs["histogram"]["channels"][0]
    assert channel["finite_samples"] == 4 and channel["outside_range"] == 2
    assert channel["statistics"] == pytest.approx({
        "minimum": 0, "maximum": 14, "mean": 6.5, "median": 6,
        "standard_deviation": np.std([0, 2, 10, 14]),
        "lower_threshold": 1.5, "upper_threshold": 11,
    })
    assert result.outputs["threshold"] == 6.5
    marked = result.outputs["plot"]
    block.parameters["show_statistics"] = False
    plain = block.execute({"image": image, "mask": mask})
    assert plain.outputs["threshold"] == 6.5
    assert np.any(marked != plain.outputs["plot"])


def test_plot_lists_all_channels_and_numeric_threshold_uses_selected_channel(monkeypatch):
    rows = {}
    put_text = cv2.putText

    def capture_text(image, text, origin, *args):
        rows.setdefault(origin[1], []).append(text)
        return put_text(image, text, origin, *args)

    monkeypatch.setattr(cv2, "putText", capture_text)
    block = builtin_registry().create(
        "inspect.histogram", parameters={"statistics_channel": "G", "threshold_statistic": "median"}
    )
    image = ImageArray(np.array([[[1, 20, 150], [3, 40, 210]]], np.uint8), "rgb")
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["mean"] == result.outputs["median"] == result.outputs["threshold"] == 30
    channels = result.outputs["histogram"]["channels"]
    assert [channel["statistics"]["mean"] for channel in channels] == [2, 30, 180]
    legend, heading, *statistics_rows = [rows[y] for y in sorted(rows)]
    assert legend == ["R", "G", "B", "Markers: G"]
    assert heading == ["Channel", "Mean", "Median", "Std dev", "Min", "Max", "P5", "P95"]
    assert statistics_rows == [
        ["R", "2", "2", "1", "1", "3", "1.1", "2.9"],
        ["G", "30", "30", "10", "20", "40", "21", "39"],
        ["B", "180", "180", "30", "150", "210", "153", "207"],
    ]

    rows.clear()
    block.parameters["show_statistics"] = False
    assert block.execute({"image": image}).outputs["threshold"] == 30
    assert list(rows.values()) == [["R", "G", "B"]]
    block.parameters["statistics_channel"] = "A"
    assert block.execute({"image": image}) is None
    assert block.error_kind == "invalid" and "unavailable" in block.error


def test_wired_thresholds_use_exact_bounds_and_wait_for_a_measurement():
    registry = builtin_registry()
    graph = Graph()
    histogram = graph.add(registry.create(
        "inspect.histogram", parameters={"lower_percentile": 20, "upper_percentile": 80}
    ))
    tail = graph.add(registry.create("cv.threshold"))
    middle = graph.add(registry.create("cv.in_range"))
    histogram.exposed_outputs.extend(("lower_threshold", "upper_threshold"))
    tail.exposed_inputs.append("value")
    middle.exposed_inputs.extend(("lower", "upper"))
    graph.connect(histogram.id, "upper_threshold", tail.id, "value")
    graph.connect(histogram.id, "lower_threshold", middle.id, "lower")
    graph.connect(histogram.id, "upper_threshold", middle.id, "upper")
    image = np.arange(7, dtype=np.uint8)[None]
    inputs = {(block.id, "image"): image for block in (histogram, tail, middle)}
    graph.run(injected=inputs)
    assert histogram.last_result.outputs["lower_threshold"] == pytest.approx(1.2)
    assert histogram.last_result.outputs["upper_threshold"] == pytest.approx(4.8)
    np.testing.assert_array_equal(tail.last_result.outputs["image"], [[0, 0, 0, 0, 0, 255, 255]])
    np.testing.assert_array_equal(
        middle.last_result.outputs["image"], [[0, 0, 255, 255, 255, 0, 0]]
    )

    inputs[histogram.id, "mask"] = np.zeros_like(image)
    graph.run(injected=inputs)
    assert histogram.last_result.outputs["threshold"] is NO_VALUE
    assert histogram.last_result.outputs["histogram"]["channels"][0]["statistics"]["mean"] is None
    assert tail.state == middle.state == "inactive"
    assert tail.current_emissions == middle.current_emissions == {}


def test_ticketed_threshold_does_not_broadcast_over_images_with_missing_statistics():
    class Burst(Block):
        outputs = (Port("image"),)

        def process(self, inputs):
            clock = TicketClock()
            image = np.arange(6, dtype=np.float32)[None]
            packets = [
                ValuePacket(frame, {"ticket": clock.issue(self.id)})
                for frame in (image, np.full_like(image, np.nan))
            ]
            return Result({"image": image}, emissions={"image": packets})

    registry = builtin_registry()
    graph = Graph()
    source = graph.add(Burst())
    histogram = graph.add(registry.create("inspect.histogram"))
    threshold = graph.add(registry.create("cv.threshold"))
    histogram.exposed_outputs.append("threshold")
    threshold.exposed_inputs.append("value")
    for block in (histogram, threshold):
        graph.connect(source.id, "image", block.id, "image")
    graph.connect(histogram.id, "threshold", threshold.id, "value")
    graph.run()
    assert len(histogram.current_emissions["threshold"]) == 1
    assert threshold.state == "error" and "Stream lengths differ" in threshold.error
    assert threshold.current_emissions == {}


def test_depth_units_pairing_sampling_and_nested_snapshots(tmp_path):
    path = tmp_path / "depth.png"
    raw = np.array([[0, 1000], [2000, 3000]], np.uint16)
    ok, encoded = cv2.imencode(".png", raw)
    assert ok
    path.write_bytes(encoded.tobytes())
    registry = builtin_registry()
    graph = Graph()
    load = graph.add(registry.create("input.depth_file", parameters={"path": str(path)}))
    sample = graph.add(
        registry.create("inspect.depth_sample", parameters={"pixel_points": [[0, 0], [1, 0]]})
    )
    graph.connect(load.id, "depth", sample.id, "depth")
    nested = encapsulate(graph, [sample.id], "Depth samples", registry)
    graph.run()
    assert load.state == "pass", load.error
    assert nested.state == "pass", nested.error
    depth = load.last_output_images["depth"]
    assert isinstance(depth, DepthArray) and depth.units == "m"
    assert np.isnan(depth[0, 0]) and depth[1, 1] == pytest.approx(3)
    assert copy.deepcopy(depth).units == "m"
    assert np.all(preview_rgb(depth)[0, 0] == 0)
    result = next(iter(nested.last_result.outputs.values()))
    assert result["samples"] == [{"pixel": [0, 0], "depth": None}, {"pixel": [1, 0], "depth": 1.0}]
    paired = registry.create("image.rgbd")
    paired.execute({"image": np.zeros((2, 2, 3), np.uint8), "depth": depth})
    assert paired.state == "pass" and isinstance(paired.last_output_images["depth"], DepthArray)
    paired.execute({"image": np.zeros((3, 2, 3), np.uint8), "depth": depth})
    assert "identical dimensions" in paired.error
    histogram = graph.add(registry.create("inspect.histogram"))
    graph.connect(load.id, "depth", histogram.id, "image")
    restored = Graph.from_dict(graph.to_dict(), registry)
    restored.run()
    assert restored.blocks[nested.id].state == "pass"
    measured = restored.blocks[histogram.id]
    assert measured.state == "pass", measured.error
    assert isinstance(measured.last_inputs["image"], DepthArray)
    assert measured.last_inputs["image"].units == "m"
