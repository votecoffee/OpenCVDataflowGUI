"""Original pixel references and reproducible tagged PNG/JPEG round-trip graphs."""

import json
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageCms

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.codec_metadata import ImageMetadata, make_exif, native_chunks
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" xmlns:df="urn:opencv-dataflow:fixture:" '
    'df:note="Original Dataflow texture"/></rdf:RDF></x:xmpmeta>'
)


def main():
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    profile[24:36] = struct.pack(">6H", 2026, 1, 1, 0, 0, 0)
    profile[84:100] = b"\0" * 16  # optional profile ID, independent of creation time
    (HERE / "srgb.icc").write_bytes(profile)
    y, x = np.indices((32, 48))
    rgba = np.stack(
        [
            (x * 1200 + y * 37) % 65536,
            (y * 1800 + 1000) % 65536,
            ((x + y) * 650) % 65536,
            10000 + x * 1000,
        ],
        axis=2,
    ).astype(np.uint16)
    bgra = rgba[..., [2, 1, 0, 3]]
    chunks = [
        (0, make_exif(6, "Dataflow original fixture", "Original controlled pixel ramp")),
        (1, XMP.encode()),
        (2, bytes(profile)),
        (3, bytes([1, 13, 0, 1])),
    ]
    packet = ImageMetadata(tuple(chunks))
    kinds, raw = native_chunks(packet, ".png", bgra)
    assert cv2.imwriteWithMetadata(str(HERE / "source.png"), bgra, kinds, raw)
    assert cv2.imwrite(str(HERE / "oriented_reference.png"), np.rot90(bgra, -1))
    truth = {
        "source_size": [48, 32],
        "oriented_size": [32, 48],
        "exif_orientation": 6,
        "scope": "Original MIT synthetic RGBA16 fixture; source ICC generated with LittleCMS",
        "png_chunk_types": ["exif", "xmp", "icc", "cicp"],
        "jpeg_chunk_types": ["exif", "xmp", "icc"],
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())
    registry = builtin_registry()
    for name in ("png_roundtrip", "jpeg_roundtrip", "oriented_once", "unsupported_jpeg_cicp"):
        jpeg = name in ("jpeg_roundtrip", "unsupported_jpeg_cicp")
        graph = Graph()
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": "source.png",
                    "path_base": "project",
                    "metadata_mode": "read",
                    "color_format": "rgb" if jpeg else "rgba",
                    "pixel_depth": "uint8" if jpeg else "unchanged",
                },
            )
        )
        source.name, source.position = "Original tagged pixel ramp", [0, 0]
        source.exposed_outputs.append("metadata")
        encoder = graph.add(
            registry.create(
                "cv.encode_image",
                block_id="encode",
                parameters={
                    "codec": "jpeg" if jpeg else "png",
                    "jpeg_quality": 95,
                    "png_compression": 7,
                },
            )
        )
        encoder.position = [360, 0]
        encoder.exposed_inputs.append("metadata")
        decoder = graph.add(
            registry.create(
                "cv.decode_image",
                block_id="decode",
                parameters={
                    "metadata_mode": "apply_orientation" if name == "oriented_once" else "read",
                },
            )
        )
        decoder.position = [720, 0]
        decoder.exposed_outputs += ["metadata", "info"]
        graph.connect("source", "image", "encode", "image")
        if jpeg:
            metadata = graph.add(
                registry.create(
                    "cv.image_metadata",
                    block_id="metadata",
                    parameters={
                        "description": "Original JPEG metadata fixture",
                        "xmp": XMP,
                        "icc_path": "srgb.icc",
                        "cicp": [1, 13, 0, 1] if name == "unsupported_jpeg_cicp" else [],
                    },
                )
            )
            metadata.position = [0, 400]
            graph.connect("metadata", "metadata", "encode", "metadata")
        else:
            graph.connect("source", "metadata", "encode", "metadata")
        graph.connect("encode", "encoded", "decode", "encoded")
        if name == "oriented_once":
            fresh = graph.add(
                registry.create(
                    "cv.image_metadata",
                    block_id="fresh",
                    parameters={
                        "description": "Fresh metadata after explicit orientation",
                        "icc_path": "srgb.icc",
                        "xmp": XMP,
                        "cicp": [1, 13, 0, 1],
                    },
                )
            )
            fresh.position = [720, 420]
            reencoder = graph.add(registry.create("cv.encode_image", block_id="reencode"))
            reencoder.position = [1080, 0]
            reencoder.exposed_inputs.append("metadata")
            graph.connect("decode", "image", "reencode", "image")
            graph.connect("fresh", "metadata", "reencode", "metadata")
        elif name == "png_roundtrip":
            reencoder = graph.add(registry.create("cv.encode_image", block_id="reencode"))
            reencoder.position = [1080, 0]
            reencoder.exposed_inputs.append("metadata")
            graph.connect("decode", "image", "reencode", "image")
            graph.connect("decode", "metadata", "reencode", "metadata")
        if name in ("oriented_once", "png_roundtrip"):
            final = graph.add(
                registry.create(
                    "cv.decode_image",
                    block_id="final",
                    parameters={
                        "metadata_mode": "apply_orientation" if name == "oriented_once" else "read",
                    },
                )
            )
            final.position = [1440, 0]
            final.exposed_outputs += ["metadata", "info"]
            graph.connect("reencode", "encoded", "final", "encoded")
        note = graph.add(registry.create("layout.comment", block_id="scope"))
        note.name, note.position, note.node_size = (
            "Attribution and codec contract",
            [0, -420],
            "large",
        )
        note.description = (
            "Original MIT synthetic pixel ramps and graph, generated by build_example.py. "
            "sRGB ICC generated with LittleCMS; header date/optional ID fixed for reproducibility. "
            "EXIF orientation 6 is a display hint; source pixels remain stored unrotated. "
            "Explicit orientation uses fresh metadata before re-encoding to avoid stale geometry. "
            "PNG keeps RGBA16 pixels exactly and round-trips EXIF/XMP/ICC/CICP; JPEG uses "
            "explicit RGB8 conversion and is lossy. Native metadata readback must match before "
            "a buffer or file is emitted. JPEG CICP intentionally fails. No customer content."
        )
        graph.save(HERE / (name + ".json"))
        graph.close()


if __name__ == "__main__":
    main()
