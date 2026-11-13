"""Original inspection illustration and direct impulse references, independent of FFT code."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    h, w = 128, 192
    y, x = np.mgrid[:h, :w]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(
        mask,
        [np.array([[20, 20], [172, 20], [172, 102], [140, 102], [140, 110], [20, 110]], np.int32)],
        255,
    )
    for center in ((48, 54), (140, 54)):
        cv2.circle(mask, center, 14, 0, -1)
    cv2.rectangle(mask, (78, 78), (113, 87), 0, -1)
    base = np.where(mask != 0, 42000 + x * 30 + y * 20, 6000 + x * 15 + y * 12)
    part = np.stack((base, base * 0.90, base * 0.80), axis=2).astype(np.uint16)
    coating = (mask != 0) & (x >= 84) & (x < 104) & (y < 60)
    part[coating, 0] -= 8000
    letters = np.zeros((h, w), np.uint8)
    cv2.putText(letters, "135", (69, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
    part[letters != 0] = 9000
    assert cv2.imwrite(str(assets / "part.png"), part[..., ::-1])
    impulse = np.zeros((17, 23, 3), np.uint16)
    positions = [(3, 5), (8, 11), (12, 17)]
    values = [16384, 32768, 49152]
    phase, magnitude = np.zeros_like(impulse), np.zeros_like(impulse)
    for channel, (position, value) in enumerate(zip(positions, values, strict=True)):
        impulse[*position, channel] = value
        phase[*position, channel] = 65535
        magnitude[0, 0, channel] = value
    for name, image in (
        ("impulse", impulse),
        ("phase", phase),
        ("magnitude", magnitude),
        ("full", impulse),
    ):
        assert cv2.imwrite(str(assets / f"{name}.png"), image[..., ::-1])
    (assets / "truth.json").write_text(
        json.dumps(
            {
                "reference": "Direct pixel assignments; expected images do not use a transform",
                "image_size": [23, 17],
                "channel_order": "RGB",
                "positions_yx": positions,
                "unsigned_amplitudes": values,
                "unsigned_maximum": 65535,
                "phase_only": "Unit impulses at original positions",
                "magnitude_only": "Original unit-range amplitudes at the zero-phase origin [0,0]",
                "maximum_error": 1e-12,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    registry = builtin_registry()

    def image_source(graph, identifier, filename, position, color="rgb"):
        block = graph.add(
            registry.create(
                "input.file",
                block_id=identifier,
                parameters={
                    "path": f"assets/{filename}.png",
                    "path_base": "project",
                    "color_format": color,
                    "pixel_depth": "unchanged",
                },
            )
        )
        block.position = position
        return block

    def note(graph, description):
        block = graph.add(registry.create("layout.comment", block_id="about"))
        block.name = "Original Fourier component fixtures — MIT"
        block.description = (
            "Original part illustration, color impulses, reference images, generator and "
            "projects under this repository's MIT license. " + description + " "
            "OpenCV DFT reference: https://docs.opencv.org/5.0/main_modules/core_array.html"
        )
        block.node_size, block.position = "large", [-330.0, -350.0]

    graph = Graph()
    try:
        image_source(graph, "source", "part", [-330.0, 0.0])
        for row, mode in enumerate(("full", "phase_only", "magnitude_only")):
            block = graph.add(
                registry.create(
                    "cv.reconstruct_fourier_image", block_id=mode, parameters={"components": mode}
                )
            )
            block.name = {
                "full": "Full reconstruction",
                "phase_only": "Phase retains structure",
                "magnitude_only": "Zero phase loses source positions",
            }[mode]
            block.position = [80.0, row * 360.0]
            graph.connect("source", "image", block.id, "image")
        note(
            graph,
            "The 16-bit bracket illustration shows component effects. Full reconstruction "
            "retains unit-range source samples; phase-only whitens reliable bins; magnitude-only "
            "uses a new zero-phase grid. Raw outputs are signed floating images. Preview contrast "
            "does not alter raw values; alpha remains on the original Source output.",
        )
        graph.save(folder / "project.json")
    finally:
        graph.close()

    graph = Graph()
    try:
        image_source(graph, "source", "impulse", [-330.0, 0.0])
        for row, (name, mode) in enumerate(
            (("full", "full"), ("phase", "phase_only"), ("magnitude", "magnitude_only"))
        ):
            block = graph.add(
                registry.create(
                    "cv.reconstruct_fourier_image", block_id=name, parameters={"components": mode}
                )
            )
            block.position = [60.0, row * 360.0]
            graph.connect("source", "image", name, "image")
            reference = image_source(graph, f"truth_{name}", name, [420.0, 1100 + row * 340.0])
            unit = graph.add(
                registry.create(
                    "cv.image_arithmetic",
                    block_id=f"unit_{name}",
                    parameters={
                        "operation": "divide",
                        "scalar": 65535.0,
                        "output_depth": "float64",
                    },
                )
            )
            unit.name, unit.position = "Reference counts / 65535", [430.0, row * 360.0]
            graph.connect(reference.id, "image", unit.id, "image_a")
            verify = graph.add(
                registry.create(
                    "inspect.image_quality",
                    block_id=f"verify_{name}",
                    parameters={
                        "metric": "max_absolute_error",
                        "check_threshold": True,
                        "threshold": 1e-12,
                    },
                )
            )
            verify.name, verify.position = f"Verify analytic {name}", [790.0, row * 360.0]
            graph.connect(name, "image", verify.id, "image_a")
            graph.connect(unit.id, "image", verify.id, "image_b")
        note(
            graph,
            "The RGB impulses have independent pixel-assignment references. Phase-only "
            "must produce unit impulses at their original positions; magnitude-only retains "
            "their amplitudes at [0,0]. Arithmetic converts reference integer counts to "
            "float64 units. Every comparison must stay below 1e-12.",
        )
        graph.save(folder / "verification.json")
    finally:
        graph.close()

    graph = Graph()
    try:
        image_source(graph, "source", "part", [-330.0, 0.0], "gray")
        forward = graph.add(
            registry.create("cv.dft", block_id="dft", parameters={"size": "optimal"})
        )
        forward.position = [30.0, 0.0]
        graph.connect("source", "image", "dft", "image")
        for row, mode in enumerate(("full", "phase_only", "magnitude_only")):
            component = graph.add(
                registry.create(
                    "cv.fourier_components", block_id=mode, parameters={"components": mode}
                )
            )
            component.position = [390.0, row * 340.0]
            inverse = graph.add(registry.create("cv.idft", block_id=f"inverse_{mode}"))
            inverse.position = [750.0, row * 340.0]
            graph.connect("dft", "spectrum", component.id, "spectrum")
            graph.connect(component.id, "spectrum", inverse.id, "spectrum")
        note(
            graph,
            "The typed monochrome path exposes magnitude, phase and phase-validity maps "
            "on frequency grids. Components retain native spectrum ordering and padded size. "
            "IDFT uses the component packet's floating depth to keep signed samples.",
        )
        graph.save(folder / "spectrum_pipeline.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
