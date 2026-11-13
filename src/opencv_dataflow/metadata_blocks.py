"""Project-authored metadata for explicit image-codec connections."""

from pathlib import Path

import cv2

from .codec_metadata import MAX_CHUNK_BYTES, ORIENTATIONS, ImageMetadata, make_exif
from .core import Block, Port, Result
from .parameter_validation import ValidationError


class CreateImageMetadata(Block):
    type_id, title = "cv.image_metadata", "Create image metadata"
    category = block_type = "Image processing"
    subtype = "Codecs"
    outputs = (Port("metadata", "any"), Port("info", "data"))
    defaults = {
        "include_exif": True,
        "exif_orientation": "normal",
        "software": "OpenCVDataflowGUI",
        "description": "",
        "xmp": "",
        "icc_path": "",
        "path_base": "project",
        "cicp": [],
    }
    parameter_choices = {
        "exif_orientation": tuple(ORIENTATIONS),
        "path_base": ("project", "working_directory"),
    }
    parameter_help = {
        "include_exif": "Create fresh EXIF fields; disable to create other chunk types only.",
        "exif_orientation": "EXIF display hint; writing does not rotate stored pixels.",
        "software": "ASCII EXIF Software tag; blank omits it.",
        "description": "ASCII EXIF ImageDescription tag; use XMP for Unicode text.",
        "xmp": "UTF-8 XML; native JPEG/PNG framing is handled automatically.",
        "icc_path": "Optional ICC profile up to 1 MiB; tags pixels without color conversion.",
        "path_base": "ICC path base: saved project or working directory.",
        "cicp": "PNG [primaries, transfer, matrix, full_range] bytes; [] omits CICP.",
    }
    process_inputs_readonly = parallel_safe = True

    def process(self, inputs):
        p = self.parameters
        for name in ("software", "description", "xmp", "icc_path"):
            if not isinstance(p[name], str):
                raise ValidationError(f"{name} must be text", name)
        if type(p["include_exif"]) is not bool:
            raise ValidationError("Include EXIF must be boolean", "include_exif")
        chunks = (
            [
                (
                    cv2.IMAGE_METADATA_EXIF,
                    make_exif(ORIENTATIONS[p["exif_orientation"]], p["software"], p["description"]),
                )
            ]
            if p["include_exif"]
            else []
        )
        if p["xmp"]:
            chunks.append((cv2.IMAGE_METADATA_XMP, p["xmp"].encode("utf8")))
        if p["icc_path"]:
            path = Path(p["icc_path"])
            if not path.is_absolute() and p["path_base"] == "project":
                if self.project_dir is None:
                    raise ValidationError(
                        "Save or open the project before using a relative ICC path"
                    )
                path = self.project_dir / path
            with path.open("rb") as source:
                raw = source.read(MAX_CHUNK_BYTES + 1)
            chunks.append((cv2.IMAGE_METADATA_ICCP, raw))
        cicp = p["cicp"]
        if (
            not isinstance(cicp, list)
            or len(cicp) not in (0, 4)
            or any(type(v) is not int or not 0 <= v <= 255 for v in cicp)
        ):
            raise ValidationError("CICP must be [] or four byte values", "cicp")
        if cicp:
            chunks.append((cv2.IMAGE_METADATA_CICP, bytes(cicp)))
        packet = ImageMetadata(tuple(chunks))
        return Result(
            {"metadata": packet, "info": packet.describe()},
            message=f"{len(chunks)} image metadata chunks",
        )
