"""Original high-depth line signals, shared grids, physical scale and masked gaps."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
W, H = 513, 64
CASES = ("pixel_common", "physical_common", "masked_decimation")


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    x = np.arange(W)
    low = 18000 + 4000 * np.cos(2 * np.pi * x / 64) + 12000 * np.exp(-(((x - 320) / 16) ** 2))
    wave = low + 2000 * np.sin(2 * np.pi * 0.30 * x)
    image = np.tile(np.rint(wave).astype(np.uint16), (H, 1))
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    (assets / "line_signal.png").write_bytes(encoded.tobytes())
    mapping = {
        "model": "planar_homography_v1",
        "pixel_to_physical": [[0.25, 0, 0], [0, 0.25, 0], [0, 0, 1]],
        "units": "mm",
        "rms_error": 0,
        "image_size": [W, H],
        "origin_pixel": [0, 0],
        "plane_only": True,
    }
    (assets / "workplane.json").write_bytes((json.dumps(mapping, indent=2) + "\n").encode())
    registry = builtin_registry()
    for name in CASES:
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="capture",
                    parameters={
                        "path": "assets/line_signal.png",
                        "path_base": "project",
                        "color_format": "gray",
                        "pixel_depth": "unchanged",
                    },
                )
            )
            source.position = [0, 0]
            source_id = "capture"
            if name == "physical_common":
                calibration = graph.add(
                    registry.create(
                        "input.calibration",
                        block_id="calibration",
                        parameters={"path": "assets/workplane.json", "path_base": "project"},
                    )
                )
                plane = graph.add(
                    registry.create(
                        "calibration.rectify_workplane",
                        block_id="plane",
                        parameters={
                            "width": W,
                            "height": H,
                            "units_per_pixel": 0.25,
                            "registration": "fixed_grid",
                        },
                    )
                )
                calibration.position, plane.position = [0, 400], [400, 0]
                graph.connect("capture", "image", "plane", "image")
                graph.connect("calibration", "profile", "plane", "mapping")
                source_id = "plane"
            if name == "masked_decimation":
                mask = graph.add(
                    registry.create(
                        "mask.shape",
                        block_id="excluded",
                        parameters={"rectangle": [220, 0, 20, H], "invert": True},
                    )
                )
                mask.position = [400, -250]
                graph.connect(source_id, "image", "excluded", "image")
            for index in (0, 1):
                profile_id, resample_id = f"path_{index}", f"resample_{index}"
                profile = graph.add(
                    registry.create(
                        "cv.path_intensity_profile",
                        block_id=profile_id,
                        parameters={
                            "vertices": [[0, 16 + 32 * index], [W - 1, 16 + 32 * index]],
                            "sample_spacing": 0.5 if index and name != "masked_decimation" else 1,
                        },
                    )
                )
                resample = graph.add(
                    registry.create(
                        "cv.resample_image_profile",
                        block_id=resample_id,
                        parameters={
                            "pitch": 1 if name == "physical_common" else 4,
                            "grid": "reference" if index else "own_extent",
                            "antialias": not (index and name == "masked_decimation"),
                            "method": "linear"
                            if index and name == "masked_decimation"
                            else "kaiser_sinc",
                        },
                    )
                )
                profile.position, resample.position = [800, 320 * index], [1200, 320 * index]
                resample.node_size = "large"
                graph.connect(source_id, "image", profile_id, "image")
                graph.connect(profile_id, "profile", resample_id, "profile")
                if index:
                    graph.connect("resample_0", "profile", resample_id, "reference")
                if name == "physical_common":
                    graph.connect("plane", "scale", resample_id, "scale")
                if name == "masked_decimation":
                    graph.connect("excluded", "mask", profile_id, "mask")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original measured-profile fixtures - MIT",
                [0, -600],
                "large",
            )
            note.description = (
                "Original MIT high-depth image, calibration fixture, generator and projects "
                "by OpenCVDataflowGUI contributors. No customer or third-party images. The "
                "two horizontal paths share intensity but have their own image coordinates. "
                "One resampler supplies the common reference distance grid. Physical common "
                "uses a declared .25 mm/px rectified-workplane scale. Masked decimation "
                "compares antialiased filtering with unfiltered sampling and preserves a "
                "taught gap. Filter support widens the invalid gap; it never invents data "
                "across it. The Float64 Kaiser implementation is original; the installed "
                "native signal resampler did not qualify the tested decimation/coordinate "
                "requirements. These synthetic profiles are engineering checks, not "
                "calibration acceptance."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original repository fixtures",
                    "image_size": [W, H],
                    "peak_center_pixel": 320,
                    "physical_units_per_pixel": 0.25,
                    "masked_columns": [220, 239],
                    "signal": (
                        "18000 + 4000*cos(2*pi*x/64) + 12000*exp(-((x-320)/16)^2) "
                        "+ 2000*sin(2*pi*0.30*x), rounded uint16"
                    ),
                    "cases": list(CASES),
                    "asset_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(assets.iterdir())
                        if p.is_file()
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
