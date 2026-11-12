import numpy as np
import pytest

from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugin_dialog import PluginDialogContext
from opencv_dataflow.pointclouds import PointCloud


class CloudSource(Block):
    type_id = "test.cloud_source"
    inputs = ()
    outputs = (Port("cloud", "cloud"),)


class CloudSink(Block):
    type_id = "test.cloud_sink"
    inputs = (Port("scene", "cloud"), Port("data", "data", required=False))
    outputs = ()


def test_cloud_snapshot_copies_latest_cloud_and_payload(app):
    graph = Graph()
    source = graph.add(CloudSource())
    target = graph.add(CloudSink())
    graph.connect(source.id, "cloud", target.id, "scene")
    context = PluginDialogContext(graph, target)
    try:
        with pytest.raises(ValueError, match="No connected cloud"):
            context.snapshot_input_cloud("scene")
        cloud = PointCloud(
            np.array([[1, 2, 3]], np.float32),
            "mm",
            "camera",
            normals=np.array([[0, 0, 1]], np.float32),
        )
        source.last_result = Result(
            {"cloud": cloud}, payloads={"cloud": {"extra": {"sample": [1]}}}
        )
        packet = context.snapshot_input_cloud("scene")
        assert packet.value is not cloud
        assert packet.value.units == "mm" and packet.value.frame == "camera"
        packet.value.points[:] = 0
        packet.value.normals[:] = 0
        packet.payload["extra"]["sample"].append(2)
        np.testing.assert_array_equal(cloud.points, [[1, 2, 3]])
        np.testing.assert_array_equal(cloud.normals, [[0, 0, 1]])
        assert source.last_result.payloads["cloud"]["extra"]["sample"] == [1]
        source.last_result = Result({"cloud": NO_VALUE})
        with pytest.raises(ValueError, match="No connected cloud"):
            context.snapshot_input_cloud("scene")
        with pytest.raises(ValueError, match="not a cloud"):
            context.snapshot_input_cloud("data")
        with pytest.raises(ValueError, match="not a data"):
            context.snapshot_input_data("scene")
    finally:
        context._close()
        graph.close()
    with pytest.raises(RuntimeError, match="closed"):
        context.snapshot_input_cloud("scene")
