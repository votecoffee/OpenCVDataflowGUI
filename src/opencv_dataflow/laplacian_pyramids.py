"""Signed multiscale bands on the existing typed image-pyramid contract."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS
from .payloads import ImagePyramid, metadata, moved_payload, pyramid_level_image, validate_payload
from .processing import image_value
from .result_graphics import ACCEPTED

_DEPTHS = (np.uint8, np.uint16, np.int16, np.float32, np.float64)
_FORMATS = ("gray", "bgr", "rgb", "bgra", "rgba")
_MODEL = "laplacian_pyramid_v1"
_LIMIT = 8_388_608


def _sizes(shape, count):
    height, width = shape[:2]
    if height * width > _LIMIT:
        raise ValidationError("Laplacian pyramid base is limited to 8,388,608 pixels")
    sizes = [(width, height)]
    for _ in range(1, count):
        if (width, height) == (1, 1):
            raise ValidationError("Requested pyramid has more levels than this image supports")
        width, height = (width + 1) // 2, (height + 1) // 2
        sizes.append((width, height))
    return sizes


def laplacian_preview(image):
    """Display signed bands about neutral gray without altering their samples."""
    values = np.asarray(image, np.float64)
    payload = metadata(image)
    if payload["pyramid"]["band_kind"] == "detail":
        peak = float(np.max(np.abs(values)))
        view = np.full(values.shape, 0.5) if peak == 0 else 0.5 + 0.5 * (values / peak)
    else:
        dtype = np.dtype(payload["laplacian_pyramid"]["source_depth"])
        if dtype.kind in "iu":
            bounds = np.iinfo(dtype)
            view = (values - bounds.min) / (float(bounds.max) - bounds.min)
        else:
            view = values
    return ImageArray(np.rint(np.clip(view, 0, 1) * 255).astype(np.uint8), image_format(image))


class BuildLaplacianPyramid(Block):
    type_id, title = "cv.laplacian_pyramid", "Laplacian image pyramid set"
    category, block_type, subtype = "Image processing", "Image processing", "Pyramids"
    inputs = (Port("image"),)
    outputs = (Port("images", "pyramid"), Port("image"), Port("info", "data"))
    defaults = {
        "levels": 4,
        "backend": "floating",
        "working_depth": "float64",
        "border": "reflect_101",
    }
    parameter_choices = {
        "backend": ("floating", "native_detail"),
        "working_depth": ("float64", "float32"),
        "border": ("reflect_101", "reflect_edge", "replicate"),
    }
    parameter_ranges = {"levels": NumericRange(1, 9, whole=True)}
    parameter_help = {
        "levels": "Total levels including the final coarse base, from 1 to 9.",
        "backend": (
            "Floating uses pyrDown/pyrUp; native_detail uses createLaplacePyr on uint8 "
            "three-channel images with signed int16 bands."
        ),
        "working_depth": (
            "Floating coefficient depth; raw sample units are retained. Native detail "
            "always uses int16."
        ),
        "border": (
            "Downsampling border. Upsampling uses OpenCV's required reflect_101. Native"
            " detail requires reflect_101 for both."
        ),
    }
    input_help = {
        "image": (
            "Gray/RGB/BGR/RGBA/BGRA uint8, uint16, int16 or float32/64. "
            "Convert HSV/Lab or split channels first."
        )
    }
    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        image = image_value(source, _DEPTHS)
        encoding = image_format(source)
        if encoding not in _FORMATS:
            raise ValidationError(
                "Laplacian bands require gray/RGB/BGR/RGBA/BGRA; convert or split HSV/Lab first"
            )
        channels = () if encoding == "gray" else (4,) if encoding in ("bgra", "rgba") else (3,)
        if image.shape[2:] != channels:
            raise ValidationError("Image channel layout does not match its declared encoding")
        sizes = _sizes(image.shape, int(p["levels"]))
        if p["backend"] == "native_detail":
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValidationError(
                    "Native detail pyramids require uint8 three-channel images", "backend"
                )
            if p["border"] != "reflect_101":
                raise ValidationError(
                    "Native detail pyramids require reflect_101 borders", "border"
                )
            bands = [item.get() for item in cv2.detail.createLaplacePyr(image, len(sizes) - 1, [])]
        else:
            if np.max(np.abs(image.astype(np.float64))) > np.finfo(p["working_depth"]).max:
                raise ValidationError("Source values exceed the selected floating working depth")
            current = image.astype(p["working_depth"])
            if not np.isfinite(current).all():
                raise ValidationError("Source values exceed the selected floating working depth")
            bands = []
            for size in sizes[:-1]:
                coarse = cv2.pyrDown(current, borderType=BORDERS[p["border"]])
                expanded = cv2.pyrUp(coarse, dstsize=size)
                bands.append(cv2.subtract(current, expanded))
                current = coarse
            bands.append(current)
        source_payload = metadata(source)
        header = {
            "model": _MODEL,
            "backend": p["backend"],
            "border": p["border"],
            "working_depth": str(bands[0].dtype),
            "source_depth": str(image.dtype),
            "source_format": encoding,
            "source_context": {
                key: copy.deepcopy(source_payload[key])
                for key in ("pyramid", "laplacian_pyramid")
                if key in source_payload
            },
        }
        levels, info = [], []
        for index, (band, size) in enumerate(zip(bands, sizes, strict=True)):
            if not np.isfinite(band).all():
                raise ValidationError("Pyramid filtering exceeded the floating-point range")
            matrix = np.diag([2.0**index, 2.0**index, 1.0])
            payload = moved_payload(source_payload, matrix, list(size))
            payload["laplacian_pyramid"] = copy.deepcopy(header)
            kind = "coarse" if index == len(bands) - 1 else "detail"
            payload["pyramid"] = {
                "representation": "laplacian",
                "band_kind": kind,
                "set_id": self.id,
                "level": index,
                "level_count": len(bands),
                "largest_size": list(sizes[0]),
                "to_largest": matrix.tolist(),
            }
            levels.append(ImageArray(band, encoding, payload))
            info.append(
                {
                    "level": index,
                    "kind": kind,
                    "size": list(size),
                    "to_largest": matrix.tolist(),
                    "minimum": float(band.min()),
                    "maximum": float(band.max()),
                }
            )
        images = ImagePyramid(tuple(levels))
        result = Result(
            {"images": images, "image": images.largest, "info": info},
            message=f"{len(levels)} signed Laplacian levels ({p['backend']})",
        )
        if self.capture_graphics:
            result.graphics["signed detail (zero is gray)"] = laplacian_preview(images.largest)
        return result


def _read(images):
    header = metadata(images.largest).get("laplacian_pyramid")
    if not isinstance(header, dict) or header.get("model") != _MODEL:
        raise ValidationError(
            "Reconstruction requires a Laplacian pyramid, not a Gaussian image set"
        )
    backend = header.get("backend")
    if (
        backend not in ("floating", "native_detail")
        or header.get("border") not in ("reflect_101", "reflect_edge", "replicate")
        or header.get("source_depth") not in tuple(np.dtype(depth).name for depth in _DEPTHS)
        or header.get("source_format") not in _FORMATS
    ):
        raise ValidationError("Invalid Laplacian pyramid header")
    expected_depth = ("int16",) if backend == "native_detail" else ("float32", "float64")
    if header.get("working_depth") not in expected_depth:
        raise ValidationError("Invalid Laplacian coefficient depth")
    if backend == "native_detail" and (
        header["source_depth"] != "uint8" or header["border"] != "reflect_101"
    ):
        raise ValidationError("Native detail needs uint8 source depth and reflect_101 borders")
    context = header.get("source_context")
    if not isinstance(context, dict) or set(context) - {"pyramid", "laplacian_pyramid"}:
        raise ValidationError("Invalid saved Laplacian source context")
    try:
        validate_payload(context)
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"Invalid saved Laplacian source context: {exc}") from exc
    sizes = _sizes(images.largest.shape, len(images.levels))
    set_id = metadata(images.largest).get("pyramid", {}).get("set_id")
    if not isinstance(set_id, str) or not set_id:
        raise ValidationError("Laplacian levels need a shared nonempty set ID")
    encoding = header["source_format"]
    channels = () if encoding == "gray" else (4,) if encoding in ("bgra", "rgba") else (3,)
    if images.largest.shape[2:] != channels:
        raise ValidationError("Laplacian channel layout does not match its encoding")
    root = metadata(images.largest)["spatial"]
    for index, (level, size) in enumerate(zip(images.levels, sizes, strict=True)):
        payload = metadata(level)
        pyramid = payload.get("pyramid", {})
        kind = "coarse" if index == len(sizes) - 1 else "detail"
        expected_matrix = np.diag([2.0**index, 2.0**index, 1.0])
        if (
            payload.get("laplacian_pyramid") != header
            or pyramid.get("representation") != "laplacian"
            or pyramid.get("band_kind") != kind
            or pyramid.get("set_id") != set_id
            or pyramid.get("level_count") != len(sizes)
            or list(level.shape[:2][::-1]) != list(size)
            or str(level.dtype) != header["working_depth"]
            or image_format(level) != header["source_format"]
            or not np.isfinite(level).all()
            or not np.array_equal(pyramid.get("to_largest"), expected_matrix)
        ):
            raise ValidationError(
                "Laplacian levels have incompatible geometry, header, samples or depth"
            )
        spatial = payload["spatial"]
        if any(spatial[key] != root[key] for key in ("root_id", "root_size")) or not np.allclose(
            spatial["to_root"], np.asarray(root["to_root"]) @ expected_matrix, atol=1e-9, rtol=0
        ):
            raise ValidationError("Laplacian levels use incompatible source coordinate frames")
    return header


class ReconstructLaplacianPyramid(Block):
    type_id, title = "cv.reconstruct_laplacian_pyramid", "Reconstruct Laplacian pyramid"
    category, block_type, subtype = "Image processing", "Image processing", "Pyramids"
    inputs = (Port("images", "pyramid"),)
    outputs = (Port("image"), Port("info", "data"))
    defaults = {"output_depth": "source", "detail_gains": []}
    parameter_choices = {"output_depth": ("source", "float32", "float64")}
    parameter_help = {
        "output_depth": (
            "Source restores the original dtype without intensity rescaling; integers "
            "round and saturate. Floating output retains raw sample units."
        ),
        "detail_gains": (
            "Floating backend only: optional largest-first list, one finite gain in "
            "[-16,16] per detail level. Empty means all ones; zero removes that band. "
            "Native detail requires unit gains."
        ),
    }
    input_help = {
        "images": (
            "Complete compatible Laplacian set. The last level is the coarse base. "
            "Gaussian sets are rejected."
        )
    }
    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        images, p = inputs["images"], self.parameters
        header = _read(images)
        values = p["detail_gains"]
        if not isinstance(values, list) or (values and len(values) != len(images.levels) - 1):
            raise ValidationError(
                "Detail gains need one entry per detail level, or an empty list", "detail_gains"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or not -16 <= value <= 16
            for value in values
        ):
            raise ValidationError("Detail gains must be finite numbers in [-16,16]", "detail_gains")
        gains = values or [1.0] * (len(images.levels) - 1)
        if header["backend"] == "native_detail":
            if any(gain != 1 for gain in gains):
                raise ValidationError(
                    "Native detail reconstruction requires unit gains; "
                    "build a floating pyramid to edit bands",
                    "detail_gains",
                )
            if images.largest.ndim != 3 or images.largest.shape[2] != 3:
                raise ValidationError("Native detail reconstruction requires three channels")
            # The native routine modifies its arguments: protect upstream histories/branches.
            copies = [cv2.UMat(np.asarray(level).copy()) for level in images.levels]
            restored = cv2.detail.restoreImageFromLaplacePyr(copies)[0].get().astype(np.float64)
        else:
            restored = np.asarray(images.levels[-1], np.float64).copy()
            for index in range(len(images.levels) - 2, -1, -1):
                level = images.levels[index]
                restored = cv2.pyrUp(restored, dstsize=(level.shape[1], level.shape[0]))
                restored += np.asarray(level, np.float64) * gains[index]
        if not np.isfinite(restored).all():
            raise ValidationError("Reconstruction exceeded the floating-point range")
        depth = header["source_depth"] if p["output_depth"] == "source" else p["output_depth"]
        dtype, clipped = np.dtype(depth), 0
        if dtype.kind in "iu":
            bounds = np.iinfo(dtype)
            rounded = np.rint(restored)
            clipped = int(np.count_nonzero((rounded < bounds.min) | (rounded > bounds.max)))
            restored = np.clip(rounded, bounds.min, bounds.max)
        elif np.max(np.abs(restored)) > np.finfo(dtype).max:
            raise ValidationError(
                "Reconstruction exceeds the requested output depth", "output_depth"
            )
        _, payload = pyramid_level_image(images, self.input_payloads["images"], 0)
        payload.pop("pyramid", None)
        payload.pop("laplacian_pyramid", None)
        payload.update(copy.deepcopy(header["source_context"]))
        info = {
            "backend": header["backend"],
            "level_count": len(images.levels),
            "output_depth": depth,
            "detail_gains": list(gains),
            "clipped_samples": clipped,
        }
        payload["laplacian_reconstruction"] = copy.deepcopy(info)
        image = ImageArray(restored.astype(dtype), header["source_format"], payload)
        result = Result(
            {"image": image, "info": info},
            message=f"Reconstructed {len(images.levels)} levels; {clipped} clipped samples",
        )
        if self.generate_graphics:
            label = graphic_object("label", ((8, 22),), ACCEPTED, text=result.message)
            result.drawings["Laplacian reconstruction"] = GraphicDrawing((label,))
        return result


LAPLACIAN_PYRAMID_BLOCKS = (BuildLaplacianPyramid, ReconstructLaplacianPyramid)
