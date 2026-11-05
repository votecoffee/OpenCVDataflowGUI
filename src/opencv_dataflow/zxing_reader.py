"""Optional multi-format barcode decoding with ZXing-C++."""

import numpy as np

from .code_readers import CodeReader, barcode_image, code_record
from .core import Port
from .selection import contains

FORMATS = (
    "AllReadable",
    "AllLinear",
    "AllMatrix",
    "AllIndustrial",
    "AllRetail",
    "AllGS1",
    "DataMatrix",
    "Code128",
    "Code39",
    "Code93",
    "ITF",
    "ITF14",
    "Codabar",
    "EANUPC",
    "EAN13",
    "EAN8",
    "UPCA",
    "UPCE",
    "DataBar",
    "QRCode",
    "MicroQRCode",
    "RMQRCode",
    "PDF417",
    "MicroPDF417",
    "Aztec",
    "MaxiCode",
)


class ReadZXingBarcode(CodeReader):
    type_id, title = "inspect.read_zxing", "Read barcodes (ZXing)"
    outputs = (*CodeReader.outputs, Port("errors", "data"))
    defaults = {
        **CodeReader.defaults,
        "format": "DataMatrix",
        "try_rotate": True,
        "try_downscale": True,
        "try_invert": True,
        "text_mode": "HRI",
        "binarizer": "LocalAverage",
        "is_pure": False,
        "ean_add_on": "Ignore",
        "report_decode_errors": False,
    }
    parameter_choices = {
        "format": FORMATS,
        "text_mode": ("HRI", "Plain", "ECI", "Escaped", "Hex", "HexECI"),
        "binarizer": ("LocalAverage", "GlobalHistogram", "FixedThreshold", "BoolCast"),
        "ean_add_on": ("Ignore", "Read", "Require"),
    }
    parameter_help = {
        "format": "Limit decoding to a specific symbology or a group of formats.",
        "try_downscale": "Also search at reduced image resolutions.",
        "text_mode": "Choose how decoded bytes are formatted as text. HRI formats GS1 data.",
        "is_pure": "Use only for an aligned image containing one isolated barcode.",
        "ean_add_on": "Control reading EAN/UPC supplementary codes.",
        "report_decode_errors": "Emit invalid detections on the errors port.",
    }

    def process(self, inputs):
        try:
            import zxingcpp
        except ImportError as exc:
            raise RuntimeError(
                "ZXing barcode reading needs the barcodes extra (pip install .[barcodes])"
            ) from exc
        search, mask = barcode_image(inputs)
        codes, errors = [], []
        if search is not None:
            p = self.parameters
            barcodes = zxingcpp.read_barcodes(
                np.ascontiguousarray(search),
                formats=getattr(zxingcpp.BarcodeFormat, p["format"]),
                try_rotate=p["try_rotate"],
                try_downscale=p["try_downscale"],
                try_invert=p["try_invert"],
                text_mode=getattr(zxingcpp.TextMode, p["text_mode"]),
                binarizer=getattr(zxingcpp.Binarizer, p["binarizer"]),
                is_pure=p["is_pure"],
                ean_add_on_symbol=getattr(zxingcpp.EanAddOnSymbol, p["ean_add_on"]),
                return_errors=p["report_decode_errors"],
            )
            for barcode in barcodes:
                position = barcode.position
                corners = [
                    (point.x, point.y)
                    for point in (
                        position.top_left,
                        position.top_right,
                        position.bottom_right,
                        position.bottom_left,
                    )
                ]
                record = code_record(barcode.text, barcode.format.name, corners)
                if not contains(mask, record["center"]):
                    continue
                record.update(
                    symbology_identifier=barcode.symbology_identifier,
                    orientation_degrees=barcode.orientation,
                    content_type=barcode.content_type.name,
                    bytes_hex=barcode.bytes.hex(),
                )
                if barcode.valid:
                    codes.append(record)
                else:
                    errors.append({**record, "error": str(barcode.error)})
        return self.code_result(inputs["image"], codes, errors=errors)
