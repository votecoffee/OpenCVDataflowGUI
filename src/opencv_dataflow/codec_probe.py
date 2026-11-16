"""Point-in-time native reader/count/writer preflight without pixel decoding."""

import hashlib
import os
import re
import stat
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

MAX_BYTES = 128 * 1024**2
WRITERS = {
    "png": ".png",
    "jpeg": ".jpg",
    "tiff": ".tiff",
    "bmp": ".bmp",
    "webp": ".webp",
    "gif": ".gif",
    "jpeg2000": ".jp2",
    "hdr": ".hdr",
    "exr": ".exr",
    "avif": ".avif",
    "jpegxl": ".jxl",
    "pbm": ".pbm",
    "pgm": ".pgm",
    "ppm": ".ppm",
    "pam": ".pam",
    "pfm": ".pfm",
    "custom": None,
}
COUNT_FLAGS = {
    "unchanged": cv2.IMREAD_UNCHANGED,
    "any_color": cv2.IMREAD_ANYCOLOR,
    "color": cv2.IMREAD_COLOR_BGR,
    "grayscale": cv2.IMREAD_GRAYSCALE,
    "gdal": cv2.IMREAD_LOAD_GDAL,
}


class ProbeImageFile(Block):
    type_id, title, category = (
        "cv.probe_image_file",
        "Probe image file and codecs",
        "Image processing",
    )
    block_type, subtype = "Image processing", "Codecs"
    inputs = (Port("path", "text", required=False, required_when_connected=True),)
    outputs = (
        Port("info", "data"),
        Port("ok", "boolean"),
        Port("reader_available", "boolean", required=False),
        Port("page_count", "integer", required=False),
        Port("writer_available", "boolean", required=False),
    )
    defaults = {
        "mode": "file_and_writer",
        "path": "",
        "path_base": "project",
        "writer_format": "png",
        "custom_extension": ".png",
        "count_flags": "unchanged",
        "maximum_file_bytes": MAX_BYTES,
        "minimum_pages": 1,
        "maximum_pages": 1024,
    }
    parameter_choices = {
        "mode": ("file", "writer", "file_and_writer"),
        "path_base": ("project", "working_directory"),
        "writer_format": tuple(WRITERS),
        "count_flags": tuple(COUNT_FLAGS),
    }
    parameter_ranges = {
        "maximum_file_bytes": NumericRange(1, MAX_BYTES, whole=True),
        "minimum_pages": NumericRange(1, 1_000_000, whole=True),
        "maximum_pages": NumericRange(1, 1_000_000, whole=True),
    }
    parameter_help = {
        "mode": "Check reader/count, encoder extension, or both. Unrequested outputs do not emit.",
        "path": "A wired Path overrides this filename. Does not acquire images or issue tickets.",
        "writer_format": (
            "Encoder registry check, not a test of permissions, sample types or codec options."
        ),
        "custom_extension": "Custom format only: one extension such as .ras, not a filename.",
        "count_flags": "Native imcount flags. GDAL behavior depends on the installed backend.",
        "maximum_file_bytes": "Bound the snapshot before native calls. Preserves original files.",
        "minimum_pages": "Inclusive native-count limit; does not guarantee pixel integrity.",
        "maximum_pages": (
            "Inclusive policy limit checked after counting. "
            "Does not bound native traversal or runtime."
        ),
    }
    input_help = {
        "path": "Optional filename text; interpreted using Path base, retaining its ticket/payload."
    }
    parallel_safe = process_inputs_readonly = True
    preview_safe = False

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["mode"] != "writer" and p["maximum_pages"] < p["minimum_pages"]:
            raise ValidationError(
                "Maximum pages must be at least Minimum pages", "minimum_pages", "maximum_pages"
            )
        if not isinstance(p["path"], str):
            raise ValidationError("Path must be filename text", "path")
        if p["mode"] != "file" and p["writer_format"] == "custom":
            value = p["custom_extension"]
            if not isinstance(value, str) or not re.fullmatch(r"\.?[A-Za-z0-9]{1,16}", value):
                raise ValidationError(
                    "Custom extension must be one suffix, e.g. .png", "custom_extension"
                )

    def process(self, inputs):
        p = self.parameters
        requested_file = p["mode"] != "writer"
        requested_writer = p["mode"] != "file"
        outputs = {
            "reader_available": NO_VALUE,
            "page_count": NO_VALUE,
            "writer_available": NO_VALUE,
        }
        diagnostics = []
        info = {
            "model": "image_codec_preflight_v1",
            "opencv_version": cv2.__version__,
            "mode": p["mode"],
            "file": None,
            "writer": None,
            "pixel_decode_verified": False,
        }

        def failed(code, message):
            diagnostics.append({"code": code, "message": str(message)[:512]})

        if requested_file:
            value = inputs.get("path", p["path"])
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                raise ValidationError("Choose a file Path or connect filename text", "path")
            path = Path(value)
            if p["path_base"] == "project" and not path.is_absolute():
                if self.project_dir is None:
                    raise ValidationError(
                        "Save or open the project before using a relative file Path", "path"
                    )
                path = self.project_dir / path
            path = path.resolve()
            file_info = {
                "path": str(path),
                "source_extension": path.suffix.lower(),
                "byte_count": None,
                "sha256": None,
                "reader_available": None,
                "native_page_count": None,
                "count_flags": p["count_flags"],
                "count_flags_value": COUNT_FLAGS[p["count_flags"]],
                "header_count_ok": False,
            }
            info["file"] = file_info
            try:
                if not stat.S_ISREG(path.stat().st_mode):
                    raise OSError("Path must be a regular file")
                with path.open("rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise OSError("Path must be a regular file")
                    file_info["byte_count"] = before.st_size
                    if before.st_size > p["maximum_file_bytes"]:
                        failed("image_file_byte_limit", "File exceeds Maximum file bytes")
                        data = None
                    else:
                        data = stream.read(int(p["maximum_file_bytes"]) + 1)
                        after = os.fstat(stream.fileno())
                        if len(data) > p["maximum_file_bytes"]:
                            failed("image_file_byte_limit", "File grew beyond Maximum file bytes")
                            data = None
                        elif (before.st_size, before.st_mtime_ns) != (
                            after.st_size,
                            after.st_mtime_ns,
                        ) or len(data) != before.st_size:
                            failed(
                                "image_file_changed",
                                "File changed during the snapshot; retry the probe",
                            )
                            data = None
                if data is not None:
                    file_info.update(
                        byte_count=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),
                        signature_prefix_hex=data[:16].hex(),
                    )
                    # Snapshot via Python supports Unicode source names; native file probes
                    # in the installed Windows wheel cannot open those names directly.
                    with TemporaryDirectory(prefix="ocv-codec-probe-") as temporary:
                        native_path = Path(temporary) / "snapshot.bin"
                        native_path.write_bytes(data)
                        try:
                            reader = bool(cv2.haveImageReader(str(native_path)))
                            outputs["reader_available"] = file_info["reader_available"] = reader
                            if reader:
                                count = int(
                                    cv2.imcount(str(native_path), COUNT_FLAGS[p["count_flags"]])
                                )
                                if count < 0:
                                    raise ValueError("Native image count must not be negative")
                                outputs["page_count"] = file_info["native_page_count"] = count
                                if count == 0:
                                    failed(
                                        "image_header_unreadable",
                                        "Signature recognized but no image headers counted",
                                    )
                                elif not p["minimum_pages"] <= count <= p["maximum_pages"]:
                                    failed(
                                        "image_page_count_out_of_range",
                                        f"Native count {count}; expected "
                                        f"{p['minimum_pages']}..{p['maximum_pages']} inclusive",
                                    )
                                else:
                                    file_info["header_count_ok"] = True
                            else:
                                failed(
                                    "image_reader_unavailable",
                                    "No reader recognized the file signature",
                                )
                        except cv2.error as exc:
                            failed("image_native_probe_error", exc)
            except FileNotFoundError as exc:
                failed("image_file_missing", exc)
            except OSError as exc:
                failed("image_file_io_error", exc)

        if requested_writer:
            extension = WRITERS[p["writer_format"]]
            if extension is None:
                extension = "." + p["custom_extension"].removeprefix(".").lower()
            available = None
            try:
                available = bool(cv2.haveImageWriter(extension))
                if not available:
                    failed(
                        "image_writer_unavailable", f"No native writer available for {extension}"
                    )
            except cv2.error as exc:
                failed("image_writer_probe_error", exc)
            if available is not None:
                outputs["writer_available"] = available
            info["writer"] = {"extension": extension, "available": available}
        info["diagnostics"] = diagnostics
        ok = not diagnostics
        outputs.update(info=info, ok=ok)
        for item in diagnostics:
            self.logger.warning(f"{item['code']}: {item['message']}")
        return Result(
            outputs,
            passed=ok,
            fail_reasons=diagnostics,
            message="Codec preflight checks passed (pixels not decoded)"
            if ok
            else "; ".join(item["code"] for item in diagnostics),
        )
