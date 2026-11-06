# Minimum spanning tree

Open [minimum_spanning_tree.json](minimum_spanning_tree.json) with
**File → Open…** and click **Run once**. The project detects five dots
and uses Delaunay triangulation to generate candidate edges. The tree
connects the dots with four edges. The overlay shows which edges it kept.

Find **Minimum spanning tree** under **Inspection → Measurements**.
Connect **Edges** as undirected pairs of zero-based node indices. Connect
**Edge weights** as a parallel list of costs, or connect **Points** to
calculate Euclidean costs. When both are wired, the supplied weights
take precedence and the points position the result graphics. The block
accepts up to 10,000 nodes and 100,000 edges. Weights are finite
double-precision numbers within ±1e100, including negative costs.
Self-loops are ignored and parallel edges use the lowest cost.

**Node count** defaults to zero, which derives the count from the
points. For a graph without points, configure or wire the count,
including any isolated nodes. **Algorithm** selects Kruskal or Prim.
**Root** sets the zero-based starting node for Prim. Equal weights can
lead to different trees with the same total cost.

**Edges** and **Edge weights** report the selected pairs and costs.
**Total weight** sums them. A disconnected graph reports **Connected**
as false and emits a failed result without a partial tree.

```powershell
opencv-dataflow-run examples/minimum_spanning_tree/minimum_spanning_tree.json
```

[OpenCV buildMST reference](https://docs.opencv.org/5.0/main_modules/geometry.html#buildmst) · [All examples](../README.md)
