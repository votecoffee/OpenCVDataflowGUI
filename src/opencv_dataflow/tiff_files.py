"""Ticketed replay of pages in a multipage TIFF file."""

import copy
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, frame
from .processing import image_value
from .replay import ReplaySource

TIFF_SIGNATURES = (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_PAGE_INDEX = 2**31 - 2


class TiffFile(ReplaySource):
    type_id, title, category = "input.tiff_file", "TIFF page sequence", "Input"
    block_type, subtype = "Input", "Files"
    outputs = (Port("image"), Port("info", "data"))
    defaults = {
        "path": "",
        "path_base": "working_directory",
        "start_page": 0,
        "end_page": -1,
        "stride": 1,
        "repeat": "stop",
        "page_ticket": True,
        "color_format": "unchanged",
        "pixel_depth": "unchanged",
    }
    parameter_choices = {
        "path_base": ("working_directory", "project"),
        "repeat": ("stop", "loop"),
        "color_format": COLOR_FORMATS,
        "pixel_depth": PIXEL_DEPTHS,
    }
    parameter_ranges = {
        "start_page": NumericRange(0, MAX_PAGE_INDEX, whole=True),
        "end_page": NumericRange(-1, MAX_PAGE_INDEX, whole=True),
        "stride": NumericRange(1, 100_000, whole=True),
    }
    parameter_help = {
        "path": "Path to a TIFF file, including a multipage TIFF.",
        "start_page": "Zero-based first page.",
        "end_page": "Exclusive end page. -1 reads until OpenCV cannot decode another page.",
        "stride": "Emit every Nth page starting at the first page.",
        "page_ticket": "Use filename and page index for automatic tickets.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def close(self):
        self._buffer = None
        self._prefetched = None
        self._path = None

    def reset_runtime(self):
        self.close()
        self._sequence = {"settings": None, "next": 0, "finished": False}

    def snapshot_runtime(self):
        return copy.deepcopy(self._sequence)

    def restore_runtime(self, snapshot):
        if snapshot != self._sequence:
            self.close()
            self._sequence = copy.deepcopy(snapshot)

    def copy_runtime_from(self, other):
        self.restore_runtime(other.snapshot_runtime())

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["end_page"] != -1 and p["end_page"] <= p["start_page"]:
            raise ValidationError(
                "End page must be later than start page", "end_page", "start_page"
            )
        if "repeat" in self.recipe_bindings:
            raise ValidationError("TIFF repeat is a run setting and cannot be recipe-bound")

    @property
    def exhausted(self):
        return (
            self.finite_source
            and self._sequence["finished"]
            and self._sequence["settings"] == {**self.parameters, **self.last_recipe_parameters}
        )

    def load_bytes(self):
        value = self.parameters["path"]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError("Select a TIFF file path", "path")
        path = Path(value)
        if self.parameters["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save or open the project before using a relative TIFF path")
            path = self.project_dir / path
        path = path.resolve()
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValidationError("TIFF file exceeds the 128 MiB replay limit")
        if not data.startswith(TIFF_SIGNATURES):
            raise ValidationError("Selected file does not have a TIFF header", "path")
        self._buffer = np.frombuffer(data, np.uint8)
        self._path = path

    def decode_page(self, index):
        if index > MAX_PAGE_INDEX:
            return None
        ok, pages = cv2.imdecodemulti(self._buffer, cv2.IMREAD_UNCHANGED, range=(index, index + 1))
        return pages[0] if ok and len(pages) == 1 else None

    def issue_source_ticket(self):
        state = self._sequence
        p = self.parameters
        if state["settings"] != p:
            self.reset_runtime()
            state = self._sequence
            state["settings"] = copy.deepcopy(p)
            state["next"] = int(p["start_page"])
        if state["finished"]:
            return None
        try:
            if self._buffer is None:
                self.load_bytes()
            index = state["next"]
            if self._prefetched is None:
                self._prefetched = self.decode_page(index)
            if self._prefetched is None and p["end_page"] == -1 and index != p["start_page"]:
                if self.finite_source:
                    self.close()
                    state["finished"] = True
                    return None
                index = state["next"] = int(p["start_page"])
                self._prefetched = self.decode_page(index)
            if self._prefetched is None:
                raise ValidationError(f"Cannot decode TIFF page {index} in {self._path.name}")
            ticket = super().issue_source_ticket()
            if p["page_ticket"] and self.ticket_settings["mode"] == "auto_number":
                ticket.update(
                    kind="string",
                    value=f"{self._path.name}#page-{index:06d}",
                    session=f"{ticket['session']}:tiff:{ticket['sequence']}",
                )
            return ticket
        except Exception:
            self.close()
            state["finished"] = True
            raise

    def process(self, inputs):
        state = self._sequence
        if self._prefetched is None:
            return Result({"image": NO_VALUE, "info": NO_VALUE}, message="TIFF sequence complete")
        p = self.parameters
        index = state["next"]
        path = self._path
        try:
            image = format_image(self._prefetched, p["color_format"], p["pixel_depth"])
            self._prefetched = None
            next_page = index + int(p["stride"])
            end = int(p["end_page"])
            state["next"] = next_page
            complete = end != -1 and next_page >= end
            if not complete and end == -1 and not self._cancelled():
                # One-page lookahead lets finite replay stop after the final successful output.
                self._prefetched = self.decode_page(next_page)
                complete = self._prefetched is None
            if complete:
                if self.finite_source:
                    self.close()
                    state["finished"] = True
                else:
                    state["next"] = int(p["start_page"])
        except Exception:
            self.close()
            state["finished"] = True
            raise

        info = {
            "filename": path.name,
            "path": str(path),
            "page_index": index,
            "width": image.shape[1],
            "height": image.shape[0],
            "color_format": image_format(image),
            "pixel_depth": image.dtype.name,
        }
        return Result(
            {"image": image, "info": info},
            payloads={
                "image": {
                    "tiff_page": info,
                    "spatial": frame([image.shape[1], image.shape[0]], f"{self.id}:page:{index}"),
                }
            },
            message=f"{path.name} Ã‚Â· page {index}",
        )


TIFF_COMPRESSIONS = {
    "none": cv2.IMWRITE_TIFF_COMPRESSION_NONE,
    "lzw": cv2.IMWRITE_TIFF_COMPRESSION_LZW,
    "deflate": cv2.IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE,
    "packbits": cv2.IMWRITE_TIFF_COMPRESSION_PACKBITS,
}
TIFF_PREDICTORS = {
    "none": cv2.IMWRITE_TIFF_PREDICTOR_NONE,
    "horizontal": cv2.IMWRITE_TIFF_PREDICTOR_HORIZONTAL,
}


class WriteTiffFile(Block):
    type_id, title, category = "output.tiff_file", "Write multipage TIFF", "Output"
    block_type, subtype = "Output", "Images"
    inputs = (Port("pages", "any"),)
    outputs = (Port("info", "data"), Port("page_count", "number"), Port("byte_count", "number"))
    defaults = {
        "path": "output.tiff",
        "path_base": "working_directory",
        "compression": "lzw",
        "predictor": "native",
        "x_dpi": 0,
        "y_dpi": 0,
    }
    parameter_choices = {
        "path_base": ("working_directory", "project"),
        "compression": tuple(TIFF_COMPRESSIONS),
        "predictor": ("native", *TIFF_PREDICTORS),
    }
    parameter_ranges = {axis: NumericRange(0, 1_000_000, whole=True) for axis in ("x_dpi", "y_dpi")}
    parameter_help = {
        "path": "Choose a new .tif or .tiff filename for this batch.",
        "compression": "Lossless compression applied to every page.",
        "predictor": "Horizontal prediction is available with LZW or Deflate.",
        "x_dpi": "Horizontal resolution. Set both DPI fields to zero to use the codec defaults.",
        "y_dpi": "Vertical resolution. Set both DPI fields to zero to use the codec defaults.",
    }
    input_help = {"pages": "Ordered image group from Batch serial entries or a compatible block."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["predictor"] == "horizontal" and p["compression"] not in ("lzw", "deflate"):
            raise ValidationError("Horizontal prediction requires LZW or Deflate", "predictor")
        if bool(p["x_dpi"]) != bool(p["y_dpi"]):
            raise ValidationError("Set both DPI values, or leave both at zero", "x_dpi", "y_dpi")

    def process(self, inputs):
        group = inputs["pages"]
        if not isinstance(group, FlowGroup):
            raise ValidationError("Connect an ordered group of images to Pages")
        p = self.parameters
        value = p["path"]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError("Choose a TIFF output path", "path")
        path = Path(value)
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise ValidationError("Use a .tif or .tiff output extension", "path")
        if p["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save or open the project before using a relative TIFF path")
            path = self.project_dir / path
        path = path.parent.resolve() / path.name
        if path.exists():
            raise FileExistsError("Choose a new path; existing TIFF files are preserved")

        images = []
        for index, packet in enumerate(group.items, 1):
            page = packet.value
            if not isinstance(page, np.ndarray):
                raise ValidationError(f"Page {index} is not an image")
            image = image_value(page, (np.uint8, np.uint16))
            if image.size == 0 or image_format(page) not in ("gray", "bgr", "rgb", "bgra", "rgba"):
                raise ValidationError(f"Page {index} needs a nonempty grayscale or RGB/BGR image")
            images.append(page)
        if sum(image.nbytes for image in images) > MAX_FILE_BYTES:
            raise ValidationError("Uncompressed TIFF pages exceed the 128 MiB batch limit")

        options = [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_COMPRESSIONS[p["compression"]]]
        if p["predictor"] != "native" and p["compression"] in ("lzw", "deflate"):
            options.extend([cv2.IMWRITE_TIFF_PREDICTOR, TIFF_PREDICTORS[p["predictor"]]])
        if p["x_dpi"]:
            options.extend(
                [
                    cv2.IMWRITE_TIFF_RESUNIT,
                    cv2.IMWRITE_TIFF_RESOLUTION_UNIT_INCH,
                    cv2.IMWRITE_TIFF_XDPI,
                    int(p["x_dpi"]),
                    cv2.IMWRITE_TIFF_YDPI,
                    int(p["y_dpi"]),
                ]
            )
        ok, encoded = cv2.imencodemulti(".tiff", [opencv_image(page) for page in images], options)
        if not ok or encoded.size == 0:
            raise ValidationError("OpenCV could not encode the TIFF pages")
        if encoded.nbytes > MAX_FILE_BYTES:
            raise ValidationError("Encoded TIFF exceeds the 128 MiB replay limit")

        # Publish only a complete file, with an atomic guard against replacing another writer.
        stream = NamedTemporaryFile(prefix=".tiff-", suffix=".tmp", dir=path.parent, delete=False)
        temporary = Path(stream.name)
        try:
            with stream:
                stream.write(memoryview(encoded).cast("B"))
            if os.name == "nt":
                temporary.rename(path)
            else:
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        info = {
            "path": str(path),
            "page_count": len(images),
            "byte_count": encoded.nbytes,
            "compression": p["compression"],
        }
        return Result(
            {"info": info, "page_count": len(images), "byte_count": encoded.nbytes},
            message=f"Saved {len(images)} pages to {path.name}",
        )
