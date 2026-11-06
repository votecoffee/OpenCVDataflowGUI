"""Find minimum-cost connections in a weighted graph."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .processing import integer
from .result_graphics import ACCEPTED
from .sample_tables import finite_features, read_sample_table

_MAX_NODES = 10_000
_ALGORITHMS = {"prim": cv2.MST_PRIM, "kruskal": cv2.MST_KRUSKAL}


def _read_edges(value, count):
    if isinstance(value, (list, tuple)) and not value:
        return np.empty((0, 2), dtype=np.int32)
    edges = read_sample_table(value, max_columns=2)
    if (
        edges.shape[1] != 2
        or np.any(edges < 0)
        or np.any(edges >= count)
        or not np.array_equal(edges, np.rint(edges))
    ):
        raise ValidationError(f"Edges must be pairs of whole node indices from 0 to {count - 1}")
    return edges.astype(np.int32)


def _read_weights(value, count):
    weights = np.asarray(value)
    if (
        weights.shape != (count,)
        or weights.dtype.kind not in "iuf"
        or not np.isfinite(weights).all()
        or np.any(weights > 1e100)
        or np.any(weights < -1e100)
    ):
        raise ValidationError("Provide one finite weight within ±1e100 per edge")
    return weights


class MinimumSpanningTree(Block):
    type_id, title = "inspect.minimum_spanning_tree", "Minimum spanning tree"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("edges", "data"),
        Port("edge_weights", "data", required=False, required_when_connected=True),
        Port("points", "data", required=False, required_when_connected=True),
        Port("node_count", "number", required=False, required_when_connected=True),
    )
    outputs = (
        Port("connected", "boolean"),
        Port("edges", "data"),
        Port("edge_weights", "data"),
        Port("total_weight", "number"),
    )
    defaults = {"node_count": 0, "algorithm": "kruskal", "root": 0}
    parameter_choices = {"algorithm": tuple(_ALGORITHMS)}
    parameter_ranges = {
        "node_count": NumericRange(0, _MAX_NODES, whole=True),
        "root": NumericRange(0, _MAX_NODES - 1, whole=True),
    }
    parameter_help = {
        "node_count": "Zero derives the count from the points. A wired count takes precedence.",
        "algorithm": "Use OpenCV's Prim or Kruskal algorithm on an undirected graph.",
        "root": "Zero-based starting node for Prim. Kruskal ignores this setting.",
    }
    process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        points = None
        if "points" in inputs:
            points = finite_features(inputs["points"], limit=10_000_000, max_columns=2)
            if points.shape[1] != 2:
                raise ValidationError("Points must contain an [x, y] pair for each node")
        count = inputs.get("node_count", p["node_count"])
        if "node_count" not in inputs and count == 0 and points is not None:
            count = len(points)
        count = integer(count, "Node count", 1, _MAX_NODES)
        if points is not None and len(points) != count:
            raise ValidationError("Node count must match the number of points", "node_count")
        require_same_frame(
            [
                ValuePacket(inputs[name], self.input_payloads.get(name, {}))
                for name in ("edges", "edge_weights", "points")
                if name in inputs
            ],
            message="Points, edges and weights must use the same coordinate frame",
        )
        edges = _read_edges(inputs["edges"], count)
        if "edge_weights" in inputs:
            weights = _read_weights(inputs["edge_weights"], len(edges))
        elif points is not None:
            weights = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        elif len(edges):
            raise ValidationError("Connect edge weights or points to define edge costs")
        else:
            weights = ()
        root = int(p["root"]) if p["algorithm"] == "prim" else 0
        if root >= count:
            raise ValidationError(f"Prim root must be between 0 and {count - 1}", "root")

        # A single node is connected by the empty tree. OpenCV returns false for an empty graph.
        connected, tree = True, ()
        if count > 1:
            native_edges = []
            for (source, target), weight in zip(edges, weights):
                edge = cv2.MSTEdge()
                edge.source, edge.target, edge.weight = int(source), int(target), float(weight)
                native_edges.append(edge)
            connected, tree = cv2.buildMST(count, native_edges, _ALGORITHMS[p["algorithm"]], root)
        source = "points" if points is not None else "edges"
        payload = self.input_payloads.get(source, {})
        if not connected:
            return Result(
                {
                    "connected": False,
                    "edges": NO_VALUE,
                    "edge_weights": NO_VALUE,
                    "total_weight": NO_VALUE,
                },
                payloads={"connected": payload},
                passed=False,
                message="The graph is disconnected",
                fail_reasons=["spanning_tree_disconnected"],
            )
        if len(tree) != count - 1:
            raise ValidationError("OpenCV returned an incomplete spanning tree")
        ordered = sorted(
            (min(edge.source, edge.target), max(edge.source, edge.target), edge.weight)
            for edge in tree
        )
        pairs = [[a, b] for a, b, _ in ordered]
        costs = [weight for _, _, weight in ordered]
        total = math.fsum(costs)
        outputs = {"connected": True, "edges": pairs, "edge_weights": costs, "total_weight": total}
        drawings = {}
        if self.generate_graphics and points is not None and pairs:
            lines = tuple(graphic_object("polyline", points[pair], ACCEPTED) for pair in pairs)
            drawings["Spanning tree"] = GraphicDrawing(lines, "points")
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            drawings=drawings,
            message=f"{count} nodes, {len(pairs)} edges, total weight {total:.6g}",
        )
