"""Export bounded, named review previews without changing numerical outputs."""

import json
from pathlib import Path

import cv2
from PIL import Image

from .assistant_api import _graphs
from .filmstrip import annotate_filmstrip_image
from .graphics_layers import graphics_bundle
from .images import preview_rgb


def export_previews(graph, directory, *, max_images=128, max_side=1024):
    """Create a new directory with final/retained previews and their provenance.

    PNGs use the same 8-bit display conversion as the editor. They are review
    images, not exports of original high-depth samples or point clouds.
    """
    if type(max_images) is not int or not 1 <= max_images <= 1024:
        raise ValueError("max_images must be from 1 to 1024")
    if type(max_side) is not int or not 32 <= max_side <= 4096:
        raise ValueError("max_side must be from 32 to 4096")
    directory = Path(directory).resolve()
    directory.mkdir()  # Exclusive: never overwrite another run's review.
    records = []
    omitted = 0
    for current, scope, _ in _graphs(graph):
        for block in current.blocks.values():
            for kind, images in (
                ("output", block.last_output_images),
                ("graphic", block.last_graphics),
            ):
                for name, source in images.items():
                    if len(records) >= max_images:
                        omitted += 1
                        continue
                    rgb = preview_rgb(source)
                    payload = (
                        block.last_result.payloads.get(name, {})
                        if block.last_result is not None and kind == "output"
                        else {}
                    )
                    frame = payload.get("spatial")
                    layers = (
                        graphics_bundle(block, block.last_result).layers
                        if block.last_result is not None and kind == "output"
                        else ()
                    )
                    if frame and layers:
                        rgb = cv2.cvtColor(
                            annotate_filmstrip_image(
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), frame, layers
                            ),
                            cv2.COLOR_BGR2RGB,
                        )
                    image = Image.fromarray(rgb)
                    image.thumbnail((max_side, max_side))
                    path = directory / f"{len(records) + 1:04d}.png"
                    image.save(path)
                    records.append(
                        {
                            "path": str(path),
                            "scope": list(scope),
                            "block": block.id,
                            "name": name,
                            "kind": kind,
                            "state": block.state,
                            "retained": not bool(block.current_emissions),
                            "payload": payload,
                            "source_shape": list(source.shape),
                            "source_dtype": str(source.dtype),
                            "display_size": list(image.size),
                        }
                    )
    manifest = {"images": records, "omitted": omitted, "display_only": True}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest
