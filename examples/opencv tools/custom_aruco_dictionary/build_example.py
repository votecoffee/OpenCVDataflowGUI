"""Build a seeded ArUco dictionary example with a controlled bit-error target."""

import argparse
from pathlib import Path

import cv2

from opencv_dataflow.aruco_dictionary import compile_dictionary, write_dictionary_file
from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SETTINGS = {
    "marker_size": 6,
    "marker_count": 24,
    "seed": 6240,
    "minimum_hamming_distance": 12,
    "max_correction_bits": 2,
}
ORDER = [7, 3, 18, 1, 22, 12]
DAMAGED_ID = 7
FLIPS = ((1, 2), (4, 5), (2, 4))


def damaged_marker(dictionary, flips=2, marker_id=DAMAGED_ID):
    pitch, border = 18, 1
    side = (dictionary.markerSize + 2 * border) * pitch
    image = dictionary.generateImageMarker(marker_id, side, border)
    for row, column in FLIPS[:flips]:
        y, x = (row + border) * pitch, (column + border) * pitch
        image[y : y + pitch, x : x + pitch] ^= 255
    return cv2.copyMakeBorder(image, 22, 22, 22, 22, cv2.BORDER_CONSTANT, value=255)


def build(directory):
    path = directory / "custom_aruco_dictionary.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    assets = directory / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    registry = builtin_registry()
    generator = registry.create("calibration.create_aruco_dictionary", parameters=SETTINGS)
    result = generator.execute({})
    if not result or not result.passed:
        raise ValueError(generator.error or "The synthetic dictionary did not qualify")
    record = result.outputs["dictionary"]
    write_dictionary_file(assets / "dictionary.json", record)
    ok, encoded = cv2.imencode(".png", damaged_marker(compile_dictionary(record)))
    if not ok:
        raise ValueError("Could not encode the reference marker image")
    (assets / "damaged_marker.png").write_bytes(encoded.tobytes())

    graph = Graph()

    def add(kind, key, name, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.name, block.position = name, pos
        if block.ticket_source:
            block.ticket_settings = {"mode": "fixed", "value": "custom-aruco"}
        return block

    try:
        note = add(
            "layout.comment", "note", "Generate and qualify a custom marker family", [0, -260]
        )
        note.node_size = "large"
        note.description = (
            "Original seeded markers and synthetic bit damage by OpenCVDataflowGUI "
            "contributors (MIT). Generate a dictionary, render a six-marker grid, "
            "then detect it using the saved dictionary. The separate marker image "
            "has two flipped data modules."
        )
        add(
            "calibration.create_aruco_dictionary",
            "generate",
            "Generate the seeded dictionary",
            [0, 0],
            **SETTINGS,
        )
        add(
            "calibration.generate_fiducial_target",
            "target",
            "Render IDs in board order",
            [355, 0],
            layout="aruco_grid",
            dictionary="custom",
            columns=3,
            rows=2,
            marker_ids=ORDER,
            marker_length=22.0,
            marker_separation=8.0,
            margin=6.0,
            dpi=127,
        )
        add(
            "input.aruco_dictionary",
            "load",
            "Load the saved codewords",
            [345, 260],
            path="assets/dictionary.json",
        )
        add(
            "input.file",
            "corrupt",
            "Marker with two flipped modules",
            [345, 504],
            path="assets/damaged_marker.png",
            path_base="project",
            color_format="gray",
        )
        add(
            "inspect.aruco_markers",
            "grid",
            "Decode all six IDs",
            [750, 50],
            dictionary="custom",
            required_ids=ORDER,
            minimum_markers=len(ORDER),
            error_correction_rate=0,
        )
        add(
            "inspect.aruco_markers",
            "correct",
            "Correct the two damaged bits",
            [750, 510],
            dictionary="custom",
            required_ids=[DAMAGED_ID],
            error_correction_rate=1.0,
        )
        for source, port, target, terminal in (
            ("generate", "dictionary", "target", "dictionary"),
            ("target", "image", "grid", "image"),
            ("load", "dictionary", "grid", "dictionary"),
            ("load", "dictionary", "correct", "dictionary"),
            ("corrupt", "image", "correct", "image"),
        ):
            graph.connect(source, port, target, terminal)
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
