"""Indexed native TIFF page decoding with immutable snapshots and bounded raw-page LRU."""

import copy
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .core import Block, Port, Result
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format
from .multipage_codec import read_page
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame
from .tiff_directory import MAX_BYTES, MAX_CONTAINER_PAGES, MAX_PIXELS, directory


def file_stamp(stat):
    # Windows stat/fstat can report different legacy ctime meanings. Compare mtime there.
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns if os.name != "nt" else 0,
    )


class TiffPageAccess(Block):
    type_id, title, category = "input.tiff_page", "Read TIFF page on demand", "Input"
    block_type, subtype = "Input", "Files"
    inputs = (
        Port("page_index", "integer", required=False, required_when_connected=True),
        Port("reset", "boolean", required=False),
    )
    outputs = (Port("image"), Port("page_count", "integer"), Port("info", "data"))
    defaults = {
        "path": "",
        "path_base": "working_directory",
        "page_index": 0,
        "page_ticket": True,
        "color_format": "unchanged",
        "pixel_depth": "unchanged",
        "refresh": "on_change",
        "cache_pages": 4,
        "cache_bytes": 16 * 1024**2,
        "maximum_encoded_bytes": MAX_BYTES,
        "maximum_page_pixels": MAX_PIXELS,
        "maximum_decoded_bytes": 256 * 1024**2,
    }
    parameter_choices = {
        "path_base": ("working_directory", "project"),
        "color_format": COLOR_FORMATS,
        "pixel_depth": PIXEL_DEPTHS,
        "refresh": ("on_change", "verify_content", "pinned"),
    }
    parameter_ranges = {
        "page_index": NumericRange(0, MAX_CONTAINER_PAGES - 1, whole=True),
        "cache_pages": NumericRange(0, 64, whole=True),
        "cache_bytes": NumericRange(0, MAX_BYTES, whole=True),
        "maximum_encoded_bytes": NumericRange(8, MAX_BYTES, whole=True),
        "maximum_page_pixels": NumericRange(1, MAX_PIXELS, whole=True),
        "maximum_decoded_bytes": NumericRange(1, 512 * 1024**2, whole=True),
    }
    parameter_help = {
        "path": "Classic/BigTIFF snapshot, up to 1024 pages; no native file handle is retained.",
        "page_index": "Zero-based page. A wired Page index takes precedence.",
        "page_ticket": "Automatic filename#page tickets with fresh acquisition identities.",
        "refresh": (
            "On change checks file identity/size/times. Verify content reads/hashes each request. "
            "Pinned ignores external file changes until reset/path change."
        ),
        "cache_pages": "Maximum retained raw pages using least-recently-used eviction. 0=off.",
        "cache_bytes": "Maximum retained raw sample bytes. Oversize pages bypass cache. 0=off.",
        "maximum_page_pixels": "Checked from the TIFF directory before native pixel decoding.",
        "maximum_decoded_bytes": (
            "Conservative raw/conversion workspace estimate, excluding retained cache, encoded "
            "snapshot, native codec internals and runtime copies. Not a hard RSS limit."
        ),
        "color_format": "Unchanged preserves native gray/BGR/BGRA order. TIFF RGB decodes as BGR.",
        "pixel_depth": "Unchanged preserves uint8/uint16/float32/float64 samples, including HDR.",
    }
    input_help = {
        "page_index": "Integer page to read now; connected indices must emit. Order is arbitrary.",
        "reset": "True discards the snapshot and LRU before this request, including Pinned mode.",
    }
    output_help = {
        "image": "Private converted page with a current ticket and content/page-bound pixel grid.",
        "page_count": "Count of inspected TIFF directories; does not mean every page was decoded.",
        "info": "Page declaration, snapshot SHA, cache hit/residency and cumulative counters.",
    }
    preview_safe = process_inputs_readonly = True
    requires_serial_execution = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def close(self):
        self._data, self._pages, self._path, self._stamp = None, (), None, None
        self._digest = ""
        self._cache, self._cached_bytes = OrderedDict(), 0
        self._hits = self._misses = self._evictions = self._loads = self._reads = 0

    def reset_runtime(self):
        self.close()

    def snapshot_runtime(self):
        return copy.deepcopy(
            {
                name: value
                for name, value in vars(self).items()
                if name
                in (
                    "_data",
                    "_pages",
                    "_path",
                    "_stamp",
                    "_digest",
                    "_cache",
                    "_cached_bytes",
                    "_hits",
                    "_misses",
                    "_evictions",
                    "_loads",
                    "_reads",
                )
            }
        )

    def restore_runtime(self, snapshot):
        for name, value in copy.deepcopy(snapshot).items():
            setattr(self, name, value)
        for value in self._cache.values():
            value.setflags(write=False)

    def copy_runtime_from(self, other):
        self.restore_runtime(other.snapshot_runtime())

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in self.parameter_ranges:
            if type(p[name]) is not int:
                raise ValidationError(f"{name} must be a JSON integer", name)
        if not isinstance(p["path"], str) or not p["path"].strip() or len(p["path"]) > 4096:
            raise ValidationError("Select a TIFF path up to 4096 characters", "path")
        if type(p["page_ticket"]) is not bool:
            raise ValidationError("Page ticket must be boolean", "page_ticket")
        if p["pixel_depth"] == "uint16" and p["color_format"] in ("hsv", "lab"):
            raise ValidationError(
                "HSV/Lab output requires uint8 or float32", "pixel_depth", "color_format"
            )

    def index(self, inputs):
        index = inputs.get("page_index", self.parameters["page_index"])
        return self.parameter_ranges["page_index"].check(index, "Page index", "page_index")

    def issue_source_ticket(self):
        index = self.index(self.last_inputs)
        ticket = super().issue_source_ticket()
        if self.parameters["page_ticket"] and self.ticket_settings["mode"] == "auto_number":
            ticket.update(
                kind="string",
                value=f"{Path(self.parameters['path']).name}#{index:06d}",
                session=f"{ticket['session']}:tiff_page:{ticket['sequence']}",
            )
        return ticket

    def drop_snapshot(self):
        self._evictions += len(self._cache)
        self._cache.clear()
        self._cached_bytes = 0
        self._data, self._pages, self._path, self._stamp = None, (), None, None
        self._digest = ""

    def ensure_snapshot(self):
        p = self.parameters
        path = Path(p["path"])
        if p["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save/open the project to resolve its TIFF Path", "path")
            path = self.project_dir / path
        path = path.resolve()
        try:
            if (
                self._path != str(path)
                or self._data is not None
                and (len(self._data) > p["maximum_encoded_bytes"])
            ):
                self.drop_snapshot()
            if self._data is not None and p["refresh"] == "pinned":
                return
            if (
                self._data is not None
                and p["refresh"] == "on_change"
                and (file_stamp(path.stat()) == self._stamp)
            ):
                return
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not 8 <= before.st_size <= p["maximum_encoded_bytes"]:
                    raise ValidationError(
                        "TIFF snapshot exceeds Maximum encoded bytes or is empty",
                        "maximum_encoded_bytes",
                    )
                data = stream.read(before.st_size + 1)
                after = os.fstat(stream.fileno())
                if len(data) != before.st_size or file_stamp(before) != file_stamp(after):
                    raise ValidationError("TIFF file changed during the snapshot", "path")
            self._reads += 1
            digest = hashlib.sha256(data).hexdigest()
            if self._data is not None and digest == self._digest:
                self._stamp = file_stamp(after)
                return
            pages = directory(data)
            self.drop_snapshot()
            self._data, self._pages, self._digest = data, pages, digest
            self._path, self._stamp = str(path), file_stamp(after)
            self._loads += 1
        except Exception:
            self.drop_snapshot()
            raise

    def trim(self, needed=0, inserting=False):
        p = self.parameters
        while self._cache and (
            len(self._cache) + int(inserting) > p["cache_pages"]
            or self._cached_bytes + needed > p["cache_bytes"]
        ):
            _, value = self._cache.popitem(last=False)
            self._cached_bytes -= value.nbytes
            self._evictions += 1

    def process(self, inputs):
        p = self.parameters
        if inputs.get("reset", False):
            self.reset_runtime()
        self.trim()
        self.ensure_snapshot()
        index = self.index(inputs)
        if index >= len(self._pages):
            raise ValidationError("Page index is outside this TIFF stack", "page_index")
        page = self._pages[index]
        if page.pixels > p["maximum_page_pixels"]:
            raise ValidationError("TIFF page exceeds Maximum page pixels", "maximum_page_pixels")
        dtype = np.dtype(page.dtype if p["pixel_depth"] == "unchanged" else p["pixel_depth"])
        if p["color_format"] in ("hsv", "lab") and dtype not in (np.uint8, np.float32):
            raise ValidationError(
                "HSV/Lab requires uint8 or float32 depth", "pixel_depth", "color_format"
            )
        channels = (
            page.channels
            if p["color_format"] == "unchanged"
            else {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4, "hsv": 3, "lab": 3}[
                p["color_format"]
            ]
        )
        estimate = 3 * page.nbytes + page.pixels * (
            24 * max(channels, page.channels) + 2 * channels * dtype.itemsize
        )
        if estimate > p["maximum_decoded_bytes"]:
            raise ValidationError(
                "TIFF page exceeds Maximum decoded workspace bytes", "maximum_decoded_bytes"
            )
        if self._cancelled():
            raise ValidationError("TIFF page request cancelled")
        hit = index in self._cache
        if hit:
            self._hits += 1
            raw = self._cache[index]
            self._cache.move_to_end(index)
        else:
            self._misses += 1
            retain = p["cache_pages"] > 0 and page.nbytes <= p["cache_bytes"]
            if retain:
                self.trim(page.nbytes, inserting=True)
            raw = read_page(self._data, index, page)
            if self._cancelled():
                raise ValidationError("TIFF page request cancelled")
            if retain:
                raw.setflags(write=False)
                self._cache[index] = raw
                self._cached_bytes += raw.nbytes
        image = format_image(raw, p["color_format"], p["pixel_depth"])
        info = {
            "filename": Path(self._path).name,
            "page_index": index,
            "page_count": len(self._pages),
            "container_sha256": self._digest,
            "encoded_bytes": len(self._data),
            "page": page.describe(),
            "color_format": image_format(image),
            "pixel_depth": image.dtype.name,
            "workspace_estimate_bytes": estimate,
            "refresh": p["refresh"],
            "cache": {
                "hit": hit,
                "stored": index in self._cache,
                "pages": list(self._cache),
                "bytes": self._cached_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "snapshot_reads": self._reads,
                "snapshot_loads": self._loads,
            },
        }
        return Result(
            {"image": image, "page_count": len(self._pages), "info": info},
            payloads={
                "image": {
                    "tiff_page": info,
                    "spatial": frame(list(page.size), f"{self.id}:{self._digest}:page:{index}"),
                }
            },
            message=f"{info['filename']} · page {index} · {'cache hit' if hit else 'decoded'}",
        )
