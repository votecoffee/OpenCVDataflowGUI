"""Original sparse masks and a deliberately different native hole-boundary model (MIT)."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
RECTANGLES = [[30, 30, 120, 100], [700, 50, 160, 80], [400, 600, 120, 120], [1100, 700, 200, 160]]
CASES = (
    "sparse_compare",
    "nested_boundaries",
    "selection_offset",
    "run_budget_guard",
    "count_failure",
)


def fixtures():
    sparse = np.zeros((1024, 1536), np.uint8)
    for x, y, w, h in RECTANGLES:
        sparse[y : y + h, x : x + w] = 255
    nested = np.zeros((19, 23), np.uint8)
    nested[:15, :15] = 255
    nested[3:10, 3:10] = 0
    nested[5:8, 5:8] = 255
    return sparse, nested


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    for name, image in zip(("sparse", "nested"), fixtures(), strict=True):
        ok, data = cv2.imencode(".png", image)
        assert ok
        (assets / f"{name}.png").write_bytes(data.tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()

        def add(kind, name, position, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = list(position)
            return block

        add(
            "input.file",
            "source",
            (0, 0),
            path=f"assets/{'nested' if case == 'nested_boundaries' else 'sparse'}.png",
            path_base="project",
            color_format="gray",
            pixel_depth="unchanged",
        )
        expected = (
            3
            if case == "nested_boundaries"
            else 1
            if case == "selection_offset"
            else 5
            if case == "count_failure"
            else 4
        )
        parameters = {
            "min_area": 0,
            "expected_count": expected,
            "draw_overlay": True,
            "draw_mask": True,
        }
        if case == "run_budget_guard":
            parameters["max_runs"] = 1
        if case == "selection_offset":
            parameters.update(region_mode="clip", offset=[500, -250])
        add("inspect.run_contours", "runs", (650, 0), **parameters)
        graph.connect("source", "image", "runs", "image")
        if case in ("sparse_compare", "nested_boundaries"):
            add(
                "inspect.contours",
                "dense",
                (650, 400),
                min_area=0,
                expected_count=expected,
                retrieval="list",
            )
            graph.connect("source", "image", "dense", "image")
            if case == "sparse_compare":
                add("cv.image_norm", "verify", (1080, 200), check_threshold=True, maximum=0.0)
                graph.connect("runs", "mask", "verify", "image")
                graph.connect("dense", "mask", "verify", "reference")
            else:
                add("inspect.contour_hierarchy", "tree", (650, 800), draw_overlay=False)
                graph.connect("source", "image", "tree", "image")
        elif case == "selection_offset":
            add("region.shape", "region", (320, 400), rectangle=[10, 10, 100, 100])
            graph.connect("source", "image", "region", "image")
            graph.connect("region", "region", "runs", "region")
        note = add("layout.comment", "scope", (0, -440))
        note.name, note.node_size = "Original masks and boundary semantics", "large"
        note.description = (
            "Original synthetic images/projects by OpenCVDataflowGUI contributors, MIT. "
            "No customer or third-party images. Sparse rectangles have known inclusive bounds; "
            "both APIs agree on their area/perimeter and filled mask, despite point order/winding "
            "and redundant vertical endpoints. The nested 23x19 fixture deliberately shows limits: "
            "native run areas 4,48,196 versus ordinary 4,62,196. Run-linked output is a flat list "
            "with no hole parents; ordinary Tree reports depths 0,1,2. Do not substitute polygon "
            "areas for foreground pixel counts. Clip/offset returns an 80x80 polygon with origin "
            "[500,-250] and bounds [530,-220,80,80] in the shifted frame. Run budget guard rejects "
            "before native extraction. Count failure deliberately requires 5 but finds 4, emitting "
            "a valid failed inspection. Rasters are enabled here for review, off by default in "
            "the tool. Native allocation savings do not eliminate runtime snapshots or outputs."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "cases": list(CASES),
        "rectangles_xywh": RECTANGLES,
        "sparse_polygon_areas": [(w - 1) * (h - 1) for x, y, w, h in RECTANGLES],
        "sparse_runs": sum(r[3] for r in RECTANGLES),
        "nested_run_areas": [4, 48, 196],
        "nested_dense_areas": [4, 62, 196],
        "nested_tree_depths": [0, 1, 2],
        "clip_shift_bounds": [530, -220, 80, 80],
        "clip_shift_area": 6241,
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.png"))
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
