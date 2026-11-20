"""Build a detector round trip for each printable fiducial layout."""

import argparse
from pathlib import Path

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

PRESETS = (
    (
        "marker",
        "Single ArUco marker",
        {"layout": "aruco_marker", "dictionary": "5x5_50", "marker_ids": [23], "border_bits": 2},
    ),
    (
        "grid",
        "ArUco grid board",
        {
            "layout": "aruco_grid",
            "columns": 5,
            "rows": 3,
            "marker_length": 18.35,
            "marker_separation": 6.2,
            "marker_ids": list(range(24, 39))[::-1],
            "dpi": 190,
        },
    ),
    (
        "charuco",
        "Legacy ChArUco board",
        {
            "layout": "charuco",
            "columns": 6,
            "rows": 4,
            "marker_ids": list(range(17, 29)),
            "legacy_pattern": True,
        },
    ),
)


def build(path):
    if path.exists():
        raise FileExistsError("Choose a new output path to preserve the saved example.")
    graph, registry = Graph(), builtin_registry()

    def add(type_id, block_id, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=block_id, parameters=parameters))
        block.position = [x, y]
        return block

    try:
        comment = add("layout.comment", "credit", 0, -220)
        comment.name = "Planar fiducial targets"
        comment.node_size = "large"
        comment.description = (
            "Patterns generated in the graph with OpenCV predefined dictionaries. "
            "Each branch verifies marker or corner IDs. Use the raw image and the physical "
            "print size from its target output when making a printed target."
        )
        for index, (key, name, options) in enumerate(PRESETS):
            y = 230 * index
            source = add(
                "calibration.generate_fiducial_target",
                f"make_{key}",
                0,
                y,
                **{"dpi": 127, **options},
            )
            source.name = name
            generated = source.execute({})
            if generated is None or not generated.passed:
                raise ValueError(source.error or "Could not generate the preset target")
            geometry = generated.outputs["target"]
            reader = {key: geometry[key] for key in ("dictionary", "border_bits")}
            if key == "charuco":
                reader.update(
                    {
                        field: geometry[field]
                        for field in (
                            "columns",
                            "rows",
                            "square_length",
                            "marker_length",
                            "marker_ids",
                            "legacy_pattern",
                            "units",
                        )
                    }
                )
                reader["minimum_corners"] = 15
                detect = add("calibration.charuco", "read_charuco", 405, y, **reader)
            else:
                reader.update(
                    required_ids=geometry["marker_ids"], minimum_markers=len(geometry["marker_ids"])
                )
                detect = add("inspect.aruco_markers", f"read_{key}", 405, y, **reader)
            graph.connect(source.id, "image", detect.id, "image")

        validated = validate_graph(graph)
        if not validated["valid"]:
            raise ValueError(validated["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("fiducial_targets.json")
    )
    build(parser.parse_args().output)
