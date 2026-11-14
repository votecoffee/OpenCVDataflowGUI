"""Original MIT aligned reflectance/illumination fixtures for S1-116."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    rng = np.random.default_rng(116)
    y, x = np.mgrid[:128, :192]
    reference = rng.integers(50, 190, x.shape).astype(np.uint8)
    gain = 1.25 + 0.15 * np.sin(x / 70)
    offset = 5 + 0.03 * y
    candidate = np.rint((reference - offset) / gain).astype(np.uint8)
    damaged = candidate.copy()
    damaged[60:64, 86:91] += 35
    color = rng.integers(32, 96, (*x.shape, 3), dtype=np.uint16) * 128
    color_reference = (color * np.array([0.75, 1.25, 1.5]) + [512, -100, 256]).astype(np.uint16)
    for name, image in {
        "reference": reference,
        "candidate": candidate,
        "damaged": damaged,
        "color": color,
        "color_reference": color_reference,
    }.items():
        ok, data = cv2.imencode(".png", image)
        assert ok
        (assets / f"{name}.png").write_bytes(data.tobytes())
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "seed": 116,
        "size": [192, 128],
        "scope": "Aligned synthetic sample-unit lighting; no camera calibration",
        "candidate_generation": "round((reference - (5 + 0.03*y)) / (1.25 + 0.15*sin(x/70)))",
        "defect_xywh": [86, 60, 5, 4],
        "defect_candidate_increment": 35,
        "expected_defect_pixels": 20,
        "color_reference_gain_bgr": [0.75, 1.25, 1.5],
        "color_reference_offset_bgr": [512, -100, 256],
        "raw_good_rmse": float(np.sqrt(np.mean((candidate.astype(float) - reference) ** 2))),
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())
    registry = builtin_registry()
    for case in ("local_lighting", "local_defect", "global_color", "insufficient_fit"):
        graph = Graph()

        def add(kind, name, position, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = list(position)
            return block

        source_name = (
            "damaged"
            if case == "local_defect"
            else "color"
            if case == "global_color"
            else "candidate"
        )
        ref_name = "color_reference" if case == "global_color" else "reference"
        for name, filename, y_pos in (("source", source_name, 0), ("reference", ref_name, 350)):
            add(
                "input.file",
                name,
                (0, y_pos),
                path=f"assets/{filename}.png",
                path_base="project",
                color_format="bgr" if case == "global_color" else "gray",
                pixel_depth="unchanged",
            )
        tool = add(
            "inspect.differential_exposure", "inspect", (660, 150), window=31, minimum_fit_pixels=8
        )
        if case == "global_color":
            tool.parameters.update(scope="global", outlier_limit=256, defect_threshold=1)
        graph.connect("source", "image", "inspect", "image")
        graph.connect("reference", "image", "inspect", "reference")
        if case == "insufficient_fit":
            add("region.shape", "small_fit", (330, -200), rectangle=[0, 0, 8, 8])
            graph.connect("source", "image", "small_fit", "image")
            graph.connect("small_fit", "mask", "inspect", "fit_mask")
        note = add("layout.comment", "scope", (0, -400))
        note.name = "Source, lighting model and expected result"
        note.node_size = "large"
        note.description = (
            "Original synthetic aligned intensity fixtures by OpenCVDataflowGUI contributors "
            "(MIT). No external/customer images or equipment I/O. Input pixels are aligned "
            "by construction; Pixel grid does not perform registration. truth.json records "
            "lighting formulas, injected defects and channel coefficients. Compare raw and "
            "adjusted residuals on the same valid inspection population. Image remains "
            "original; Adjusted preserves source depth/encoding; contours highlight defects. "
            "This is empirical appearance compensation, not physical radiometric calibration. "
            "Large defects or texture changes can be learned as illumination; choose known-good "
            "fit regions and an appropriate local window."
        )
        note.description += {
            "local_lighting": (
                " EXPECT PASS: smooth gain/offset variation and integer quantization, with no "
                "defect. Raw RMSE is about 38 DN; adjusted RMSE should be below 1 DN."
            ),
            "local_defect": (
                " EXPECT FAIL: 20 pixels at [86,60,5,4] are increased by 35 DN. Tukey fitting "
                "retains all 20 as adjusted-residual defects while correcting the background."
            ),
            "global_color": (
                " EXPECT PASS: exact 16-bit BGR channel gain/offset changes. Gains [.75,1.25,1.5], "
                "offsets [512,-100,256]; per-channel adjusted residuals should approach zero."
            ),
            "insufficient_fit": (
                " EXPECT FAIL: an 8x8 corner fit region cannot support local correction across "
                "the entire image. Coverage fails; unsupported pixels retain original samples "
                "and have zero adjusted residual, so always consume Valid mask."
            ),
        }[case]
        graph.save(HERE / f"{case}.json")
        graph.close()


if __name__ == "__main__":
    build()
