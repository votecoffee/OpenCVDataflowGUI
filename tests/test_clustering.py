"""Native K-means, plot output and image-space selection behavior."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def test_numeric_kmeans_centers_labels_and_plot():
    block = builtin_registry().create("ml.kmeans", parameters={"clusters": 2, "attempts": 1})
    samples = [[0, 0], [2, 0], [10, 10], [12, 10]]
    result = block.execute({"samples": samples, "initial_labels": [1, 1, 0, 0]})
    assert result is not None, block.error
    assert result.outputs["labels"] == [0, 0, 1, 1]
    assert result.outputs["centers"] == [[1.0, 0.0], [11.0, 10.0]]
    assert result.outputs["counts"] == [2, 2]
    assert result.outputs["compactness"] == pytest.approx(4)
    plot = result.outputs["plot"]
    assert plot.shape == (480, 720, 3)
    assert np.any(plot[..., 0] != plot[..., 1])
    assert result.payloads["plot"]["spatial"]["root_id"] == f"{block.id}:plot"


def test_single_row_and_one_dimensional_samples():
    block = builtin_registry().create("ml.kmeans", parameters={"clusters": 1})
    result = block.execute({"samples": [[1, 2, 3]]})
    assert result.outputs["centers"] == [[1.0, 2.0, 3.0]]
    assert result.outputs["labels"] == [0]
    assert result.outputs["compactness"] == 0
    result = block.execute({"samples": [[1], [3]]})
    assert result.outputs["centers"] == [[2.0]]
    assert result.outputs["plot"].shape == (480, 720, 3)
    block.parameters["draw_plot"] = False
    assert block.execute({"samples": [[1], [3]]}).outputs["plot"] is NO_VALUE


@pytest.mark.parametrize(
    "inputs,parameters,message",
    [
        ({"samples": [[1, 2], [3]]}, {}, "rectangular numeric table"),
        ({"samples": [[1], [2]]}, {"clusters": 3}, "at least 3 samples"),
        ({"samples": [[-(10**18)], [1]]}, {"clusters": 2}, "within ±1e15"),
        ({"samples": [[1], [2]], "initial_labels": [0, 2]}, {"clusters": 2}, "Initial labels"),
        ({"samples": [[1, 2], [3, 4]]}, {"clusters": 2, "plot_x": 2}, "Plot axes"),
    ],
)
def test_invalid_clustering_inputs(inputs, parameters, message):
    block = builtin_registry().create("ml.kmeans", parameters=parameters)
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and message in block.error


def test_image_clustering_preserves_selection_alpha_and_payload():
    block = builtin_registry().create(
        "inspect.kmeans_image", parameters={"clusters": 2, "attempts": 1}
    )
    image = np.full((4, 6, 4), 123, dtype=np.uint16)
    image[:, 1:3, :3] = (1000, 2000, 3000)
    image[:, 3:5, :3] = (8000, 9000, 10000)
    image[1, 1, 0] += 8
    image[2, 3, 1] += 8
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, 1:5] = 255
    initial = np.full(mask.shape, -1, np.int16)
    initial[:, 1:3], initial[:, 3:5] = 1, 0
    ticket = TicketClock().issue("camera")
    context = {"ticket": ticket, "spatial": frame([6, 4], "camera")}
    result = block.execute(
        {
            "image": ValuePacket(ImageArray(image, "rgba"), context),
            "mask": mask,
            "initial_labels": initial,
        }
    )
    assert result is not None, block.error
    assert result.outputs["centers"] == [[1001, 2000, 3000], [8000, 9001, 10000]]
    assert result.outputs["counts"] == [8, 8]
    labels = result.outputs["labels"]
    assert labels.dtype == np.int16
    assert np.all(labels[mask == 0] == -1)
    assert np.all(labels[:, 1:3] == 0) and np.all(labels[:, 3:5] == 1)
    mapped = result.outputs["cluster_map"]
    assert not mapped[mask == 0].any()
    assert not np.array_equal(mapped[0, 1], mapped[0, 3])
    quantized = result.outputs["quantized"]
    assert quantized.dtype == np.uint16 and image_format(quantized) == "rgba"
    np.testing.assert_array_equal(quantized[mask == 0], image[mask == 0])
    np.testing.assert_array_equal(quantized[..., 3], image[..., 3])
    for port in ("labels", "cluster_map", "quantized"):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == context["spatial"]
