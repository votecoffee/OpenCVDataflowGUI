"""Original MIPI byte fixtures, reference ADC counts and metadata-aware graphs."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def write_packed(image, bits, path, header, padding):
    group, shift = (4, 2) if bits == 10 else (2, 4)
    data = bytearray(header)
    for row in image:
        for start in range(0, len(row), group):
            samples = [int(value) for value in row[start : start + group]]
            data.extend(value >> shift for value in samples)
            data.append(
                sum(
                    (value & ((1 << shift) - 1)) << (index * shift)
                    for index, value in enumerate(samples)
                )
            )
        data.extend([0xEE] * padding)
    path.write_bytes(data)


def make_project(folder, bits, width, height, offset, padding, row_phase, column_phase):
    graph, registry = Graph(), builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.raw_sensor",
                block_id="source",
                parameters={
                    "path": f"assets/fixture.raw{bits}",
                    "path_base": "project",
                    "packing": f"mipi_raw{bits}",
                    "width": width,
                    "height": height,
                    "row_stride": width * bits // 8 + padding,
                    "byte_offset": offset,
                    "row_offset": row_phase,
                    "column_offset": column_phase,
                },
            )
        )
        reference = graph.add(
            registry.create(
                "input.file",
                block_id="reference",
                parameters={
                    "path": f"assets/reference-raw{bits}.png",
                    "path_base": "project",
                    "color_format": "gray",
                    "pixel_depth": "unchanged",
                },
            )
        )
        reference.position = [0.0, 280.0]
        verify = graph.add(
            registry.create(
                "inspect.image_quality",
                block_id="verify",
                parameters={
                    "metric": "max_absolute_error",
                    "check_threshold": True,
                    "threshold": 0.0,
                },
            )
        )
        verify.name = "Verify exact ADC counts"
        verify.position = [600.0, 270.0]
        graph.connect(source.id, "image", verify.id, "image_a")
        graph.connect(reference.id, "image", verify.id, "image_b")
        crop = graph.add(
            registry.create(
                "cv.crop",
                block_id="crop",
                parameters={
                    "x": 1,
                    "y": 1,
                    "width": width - 4,
                    "height": height - 4,
                },
            )
        )
        crop.name = "Odd-origin crop changes Bayer phase"
        crop.position = [300.0, 0.0]
        graph.connect(source.id, "image", crop.id, "image")
        demosaic = graph.add(
            registry.create(
                "cv.demosaic_bayer",
                block_id="demosaic",
                parameters={
                    "use_sensor_metadata": True,
                    "color_format": "rgb",
                },
            )
        )
        demosaic.position = [600.0, 0.0]
        graph.connect(crop.id, "image", demosaic.id, "image")
        note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
        note.name = "Original packed sensor fixtures — MIT"
        note.description = (
            "Original synthetic assets, generator and projects under the MIT license. "
            "Run once: unpacked ADC counts must match the reference with zero error. "
            "The Bayer block uses captured phase metadata and follows the odd-origin crop. "
            "Previews stretch counts for viewing; outputs retain right-aligned uint16 ADC counts. "
            "Packing reference: Linux V4L2 packed 10/12-bit Bayer documentation at docs.kernel.org."
        )
        note.node_size, note.position = "large", [0.0, -300.0]
        graph.save(folder / f"raw{bits}.json")
    finally:
        graph.close()


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    for bits, width, height, row_phase, column_phase in (
        (10, 96, 64, "odd", "even"),
        (12, 98, 65, "even", "odd"),
    ):
        y, x = np.mgrid[:height, :width]
        color = np.dstack(
            (
                0.12 + x / width * 0.2,
                0.15 + y / height * 0.2,
                0.2 + (x + y) / (width + height) * 0.15,
            )
        )
        cv2.rectangle(color, (12, 10), (width - 12, height - 10), (0.45, 0.6, 0.8), -1)
        cv2.circle(color, (width // 3, height // 2), 9, (0.03, 0.08, 0.12), -1)
        cv2.circle(color, (2 * width // 3, height // 2), 9, (0.03, 0.08, 0.12), -1)
        tile = "RGGB"
        channels = {"B": 0, "G": 1, "R": 2}
        oy, ox = int(row_phase == "odd"), int(column_phase == "odd")
        mosaic = np.array(
            [
                [
                    color[cy, cx, channels[tile[2 * ((cy + oy) % 2) + (cx + ox) % 2]]]
                    for cx in range(width)
                ]
                for cy in range(height)
            ]
        )
        mosaic = np.rint(mosaic * ((1 << bits) - 1)).astype(np.uint16)
        mosaic[0, 0], mosaic[-1, -1] = 0, (1 << bits) - 1
        header, padding = f"OCVRAW{bits}\r\n".encode("ascii"), 5
        write_packed(mosaic, bits, assets / f"fixture.raw{bits}", header, padding)
        ok, encoded = cv2.imencode(".png", mosaic)
        if not ok:
            raise RuntimeError("Unable to encode the reference sensor counts")
        (assets / f"reference-raw{bits}.png").write_bytes(encoded.tobytes())
        make_project(folder, bits, width, height, len(header), padding, row_phase, column_phase)


if __name__ == "__main__":
    main()
