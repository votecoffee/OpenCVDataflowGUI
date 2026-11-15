"""Ordered, typed TIFF page groups and raw byte transport through graph ports."""

from .core import Block, Port, Result
from .float_codecs import TIFF_COMPRESSIONS, TIFF_PREDICTORS
from .images import COLOR_FORMATS, PIXEL_DEPTHS
from .multipage_codec import decode, encode
from .parameter_validation import NumericRange, ValidationError
from .tiff_directory import MAX_BYTES, MAX_CONTAINER_PAGES, MAX_PAGES, MAX_PIXELS


class EncodePages(Block):
    type_id, title, category = (
        "cv.encode_pages",
        "Encode multipage TIFF in memory",
        "Image processing",
    )
    block_type, subtype = "Image processing", "Codecs"
    inputs = (Port("pages", "any"),)
    outputs = (
        Port("encoded", "any"),
        Port("buffer", "any"),
        Port("page_count", "integer"),
        Port("byte_count", "integer"),
        Port("info", "data"),
    )
    defaults = {
        "compression": "none",
        "predictor": "native",
        "rows_per_strip": 0,
        "x_dpi": 0,
        "y_dpi": 0,
    }
    parameter_choices = {
        "compression": tuple(TIFF_COMPRESSIONS),
        "predictor": ("native", *TIFF_PREDICTORS),
    }
    parameter_ranges = {
        "rows_per_strip": NumericRange(0, MAX_PIXELS, whole=True),
        **{axis: NumericRange(0, 1_000_000, whole=True) for axis in ("x_dpi", "y_dpi")},
    }
    parameter_help = {
        "compression": (
            "Requested lossless compression; Info reports the actual mode per page. "
            "The installed encoder uses None for float64 pages."
        ),
        "predictor": "LZW/Deflate only; horizontal needs integers, floating needs floats.",
        "rows_per_strip": "Zero uses the native strip height; otherwise set the strip height.",
        "x_dpi": "Set both DPI fields, or both zero. DPI does not set physical calibration.",
        "y_dpi": "Set both DPI fields, or both zero for codec defaults.",
    }
    input_help = {
        "pages": "Ordered FlowGroup of 1..64 gray/RGB(A)/BGR(A) images. Mixed sizes/depths allowed."
    }
    output_help = {
        "encoded": "TIFF packet with page labels, color, tickets and coordinate payloads.",
        "buffer": "Raw TIFF bytes only. Per-page application metadata travels on Encoded.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["predictor"] != "native" and p["compression"] not in ("lzw", "deflate"):
            raise ValidationError(
                "Explicit prediction requires LZW or Deflate", "predictor", "compression"
            )
        if bool(p["x_dpi"]) != bool(p["y_dpi"]):
            raise ValidationError("Set both DPI values, or leave both at zero", "x_dpi", "y_dpi")

    def process(self, inputs):
        packet = encode(inputs["pages"], self.parameters)
        info = packet.describe()
        count = len(info["pages"])
        info.update(
            page_count=count,
            requested_compression=self.parameters["compression"],
            exact_samples=True,
        )
        return Result(
            {
                "encoded": packet,
                "buffer": packet.buffer,
                "page_count": count,
                "byte_count": len(packet.buffer),
                "info": info,
            },
            message=f"Encoded and verified {count} TIFF pages",
        )


class DecodePages(Block):
    type_id, title, category = (
        "cv.decode_pages",
        "Decode multipage TIFF from memory",
        "Image processing",
    )
    block_type, subtype = "Image processing", "Codecs"
    inputs = (Port("encoded", "any"),)
    outputs = (
        Port("pages", "any"),
        Port("image"),
        Port("page_count", "integer"),
        Port("info", "data"),
    )
    defaults = {
        "start_page": 0,
        "page_count": 0,
        "stride": 1,
        "preview_page": 0,
        "color_format": "unchanged",
        "pixel_depth": "unchanged",
        "maximum_pixels": MAX_PIXELS,
        "maximum_decoded_bytes": MAX_BYTES,
    }
    parameter_choices = {"color_format": COLOR_FORMATS, "pixel_depth": PIXEL_DEPTHS}
    parameter_ranges = {
        "start_page": NumericRange(0, MAX_CONTAINER_PAGES - 1, whole=True),
        "page_count": NumericRange(0, MAX_PAGES, whole=True),
        "stride": NumericRange(1, MAX_CONTAINER_PAGES, whole=True),
        "preview_page": NumericRange(0, MAX_PAGES - 1, whole=True),
        "maximum_pixels": NumericRange(1, MAX_PIXELS, whole=True),
        "maximum_decoded_bytes": NumericRange(1, MAX_BYTES, whole=True),
    }
    parameter_help = {
        "start_page": "Zero-based page in the container.",
        "page_count": "Zero selects remaining strided pages, up to 64. Otherwise cap the count.",
        "stride": "Select every nth page; unselected pages are not decoded into pixel arrays.",
        "preview_page": "Zero-based position within the selected subset shown by Image.",
        "color_format": "Unchanged restores typed logical order; raw bytes use native BGR/BGRA.",
        "pixel_depth": "Unchanged preserves uint8/16 and float32/64; conversions use full scale.",
        "maximum_pixels": "Total selected pixels, checked before native pixel decoding.",
        "maximum_decoded_bytes": (
            "Bound for selected sample/output buffers and per-page conversion workspace. "
            "Excludes encoded bytes, runtime snapshots and native codec internals."
        ),
    }
    input_help = {
        "encoded": (
            "EncodedPages packet, TIFF bytes or a uint8 vector. Top-left contiguous gray/RGB(A), "
            "uint8/16 or float32/64; none/LZW/Deflate/PackBits."
        )
    }
    output_help = {
        "pages": "Ordered group with per-page tickets and coordinates; connect to Unbatch.",
        "image": "Selected preview page with that page's payload, not a new acquisition ticket.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True

    def process(self, inputs):
        group, info = decode(
            inputs["encoded"], self.parameters, self.input_payloads.get("encoded", {}), self.id
        )
        preview = group.items[int(self.parameters["preview_page"])]
        return Result(
            {"pages": group, "image": preview.value, "page_count": len(group.items), "info": info},
            payloads={"image": preview.payload},
            message=f"Decoded {len(group.items)} of {info['total_pages']} TIFF pages",
        )


MULTIPAGE_BLOCKS = (EncodePages, DecodePages)
