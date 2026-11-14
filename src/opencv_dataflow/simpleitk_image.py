"""Optional medical images and an explicit OpenCV plane/physical-geometry bridge."""

from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import dataclass, field

import numpy as np

from .images import ImageArray, image_format
from .parameter_validation import ValidationError
from .payloads import check_spatial, frame, metadata

INSTALL = (
    'Install the SimpleITK extra in the app environment: python -m pip install -e ".[simpleitk]"'
)
MAX_SAMPLES = 32_000_000
MAX_BYTES = 256 * 1024 * 1024
META_MODEL = "simpleitk_metadata_v1"
BRIDGE_MODEL = "simpleitk_bridge_v1"


def sitk_module():
    try:
        import SimpleITK
    except ImportError as exc:
        raise RuntimeError(INSTALL) from exc
    return SimpleITK


def pixel_types():
    sitk = sitk_module()
    names = (
        "UInt8",
        "Int8",
        "UInt16",
        "Int16",
        "UInt32",
        "Int32",
        "UInt64",
        "Int64",
        "Float32",
        "Float64",
    )
    return {
        getattr(sitk, "sitk" + prefix + name): np.dtype(name.lower())
        for prefix in ("", "Vector")
        for name in names
    }


def check_allocation(size, components, itemsize):
    if (
        len(size) not in (2, 3)
        or any(type(n) is not int or not 1 <= n <= 65536 for n in size)
        or not 1 <= components <= 16
        or math.prod(size) * components > MAX_SAMPLES
        or math.prod(size) * components * itemsize > MAX_BYTES
    ):
        raise ValidationError(
            "SimpleITK supports 2D/3D real images, up to 16 components, 32M samples and 256 MiB"
        )


def validate_native(image):
    sitk = sitk_module()
    if not isinstance(image, sitk.Image):
        raise ValidationError("Expected a SimpleITK Image")
    dtype = pixel_types().get(image.GetPixelID())
    if dtype is None:
        raise ValidationError("Convert complex or run-length-label images to real scalar pixels")
    check_allocation(list(image.GetSize()), image.GetNumberOfComponentsPerPixel(), dtype.itemsize)


def geometry(image):
    return {
        "dimension": image.GetDimension(),
        "size_xyz": list(image.GetSize()),
        "origin_xyz": list(image.GetOrigin()),
        "spacing_xyz": list(image.GetSpacing()),
        "direction_xyz": list(image.GetDirection()),
    }


def image_metadata(image):
    validate_native(image)
    result = {
        "model": META_MODEL,
        **geometry(image),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "components": image.GetNumberOfComponentsPerPixel(),
        "tags": {key: image.GetMetaData(key) for key in image.GetMetaDataKeys()},
    }
    validate_metadata(result)
    return result


def validate_metadata(document):
    if not isinstance(document, dict) or document.get("model") != META_MODEL:
        raise ValidationError("Expected simpleitk_metadata_v1 metadata")
    n = document.get("dimension")
    size = document.get("size_xyz")
    if type(n) is not int or n not in (2, 3) or not isinstance(size, list) or len(size) != n:
        raise ValidationError("Medical geometry must declare 2D or 3D size in XYZ order")
    check_allocation(size, 1, 1)
    try:
        origin = np.asarray(document.get("origin_xyz"), dtype=float)
        spacing = np.asarray(document.get("spacing_xyz"), dtype=float)
        direction = np.asarray(document.get("direction_xyz"), dtype=float).reshape(n, n)
        valid = (
            origin.shape == spacing.shape == (n,)
            and np.isfinite(origin).all()
            and np.isfinite(spacing).all()
            and np.all(spacing > 0)
            and np.isfinite(direction).all()
            and np.allclose(direction.T @ direction, np.eye(n), rtol=0, atol=1e-6)
        )
    except ValueError, TypeError:
        valid = False
    if not valid:
        raise ValidationError(
            "Medical geometry requires finite origin, positive spacing and "
            "orthonormal direction columns (reflections are allowed)"
        )
    tags = document.get("tags", {})
    if (
        not isinstance(tags, dict)
        or len(tags) > 4096
        or any(
            not isinstance(k, str)
            or not isinstance(v, str)
            or not k
            or "\0" in k
            or len(k) > 1024
            or len(v) > 65536
            for k, v in tags.items()
        )
        or len(json.dumps(tags, allow_nan=False).encode("utf-8")) > 1024 * 1024
    ):
        raise ValidationError("Medical tags must be bounded string keys/values (up to 1 MiB)")
    return document


def set_information(image, document, *, copy_geometry=True):
    validate_metadata(document)
    if copy_geometry:
        if list(image.GetSize()) != document["size_xyz"]:
            raise ValidationError("Metadata geometry size must match the actual image")
        image.SetOrigin(document["origin_xyz"])
        image.SetSpacing(document["spacing_xyz"])
        image.SetDirection(document["direction_xyz"])
    # CopyInformation intentionally omits the metadata dictionary.
    for key, value in document.get("tags", {}).items():
        image.SetMetaData(key, value)


@dataclass
class MedicalImage:
    """A native 2D/3D image that cannot accidentally enter an OpenCV image input."""

    image: object
    metadata: dict = field(default_factory=dict)
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        validate_native(self.image)
        if not self.image.IsUnique():
            # In SimpleITK 2.5.7 MakeUnique/GetArrayFromImage can discard the
            # dictionary while detaching a shared native buffer. Save it first.
            tags = {key: self.image.GetMetaData(key) for key in self.image.GetMetaDataKeys()}
            self.image.MakeUnique()
            for key, value in tags.items():
                self.image.SetMetaData(key, value)

    def __deepcopy__(self, memo):
        # The constructor detaches the copy and restores its dictionary explicitly.
        result = MedicalImage(
            sitk_module().Image(self.image), copy.deepcopy(self.metadata, memo), self.identity
        )
        memo[id(self)] = result
        return result

    def describe(self):
        return {
            "model": "simpleitk_image_v1",
            **geometry(self.image),
            "pixel_type": self.image.GetPixelIDTypeAsString(),
            "components": self.image.GetNumberOfComponentsPerPixel(),
            "tag_count": len(self.image.GetMetaDataKeys()),
        }


def require_medical(value):
    if not isinstance(value, MedicalImage):
        raise ValidationError("Use OpenCV to SimpleITK to supply a medical image")
    value.__post_init__()
    return value.image


def to_opencv(value, *, axis="z", index=0, component_mode="scalar_only", component=0):
    native = require_medical(value)
    sitk = sitk_module()
    document = image_metadata(native)
    n = native.GetDimension()
    axis_id = "xyz".index(axis) if n == 3 else None
    if type(index) is not int or index < 0 or (n == 3 and index >= native.GetSize()[axis_id]):
        raise ValidationError("Slice index is outside the volume")
    if n == 2 and index != 0:
        raise ValidationError("A 2D image only has slice index zero")
    axes = [i for i in range(n) if i != axis_id]
    if n == 3:
        size, start = list(native.GetSize()), [0] * 3
        size[axis_id], start[axis_id] = 1, index
        plane = sitk.Extract(native, size, start)  # Keep all 3 physical axes, including obliquity.
        array = np.squeeze(sitk.GetArrayFromImage(plane), axis=2 - axis_id)
    else:
        array = sitk.GetArrayFromImage(native)
    components = native.GetNumberOfComponentsPerPixel()
    selected_component = None
    encoding = "gray"
    is_vector = "vector" in native.GetPixelIDTypeAsString()
    if is_vector:
        if array.ndim == 2:  # SimpleITK squeezes a one-component vector's last axis.
            array = array[..., None]
        if component_mode == "component":
            if type(component) is not int or not 0 <= component < components:
                raise ValidationError("Component index is outside the vector image")
            array = array[..., component]
            selected_component = component
        elif component_mode in ("rgb", "rgba") and components == len(component_mode):
            encoding = component_mode
        else:
            raise ValidationError(
                "Choose RGB/RGBA explicitly for color vectors, or select a component"
            )
    elif component_mode != "scalar_only":
        raise ValidationError("Use scalar_only for a scalar medical image")
    anchor = frame(
        [array.shape[1], array.shape[0]],
        f"simpleitk:{value.identity}:{axis_id}:{index}:{selected_component}",
    )
    bridge = {
        "model": BRIDGE_MODEL,
        "image": document,
        "source_id": value.identity,
        "slice_axis": axis_id,
        "slice_index": index,
        "plane_axes": axes,
        "selected_component": selected_component,
        "root_spatial": copy.deepcopy(anchor),
    }
    payload = copy.deepcopy(value.metadata)
    payload.pop("pyramid", None)
    payload.update(simpleitk=bridge, spatial=anchor)
    return ImageArray(array, encoding, payload), bridge, payload


def from_opencv(image, document=None, *, assume_same_grid=False):
    sitk = sitk_module()
    payload = metadata(image)
    inherited = payload.get("simpleitk")
    document = copy.deepcopy(document if document is not None else inherited)
    if not isinstance(document, dict) or document.get("model") != BRIDGE_MODEL:
        raise ValidationError("Supply SimpleITK bridge metadata from SimpleITK to OpenCV")
    source = validate_metadata(document.get("image"))
    if inherited is not None and inherited != document:
        raise ValidationError("Explicit bridge metadata differs from the image's attached metadata")
    n = source["dimension"]
    axes, axis = document.get("plane_axes"), document.get("slice_axis")
    expected_axes = [i for i in range(n) if i != axis]
    if (
        axes != expected_axes
        or len(axes) != 2
        or (n == 2 and axis is not None)
        or (n == 3 and (type(axis) is not int or axis not in (0, 1, 2)))
    ):
        raise ValidationError("Invalid medical slice axes")
    index = document.get("slice_index")
    if (
        type(index) is not int
        or index < 0
        or (n == 2 and index != 0)
        or (n == 3 and index >= source["size_xyz"][axis])
    ):
        raise ValidationError("Invalid medical slice index")
    array = np.asarray(image)
    if (
        array.ndim not in (2, 3)
        or not array.size
        or array.dtype.kind not in "iuf"
        or (array.ndim == 3 and array.shape[2] not in (3, 4))
    ):
        raise ValidationError("Expected a real monochrome or explicit RGB/BGR(A) OpenCV image")
    if array.ndim == 3:
        encoding = image_format(image)
        if encoding in ("bgr", "bgra"):
            array = array[..., [2, 1, 0] + ([3] if encoding == "bgra" else [])]
        elif encoding not in ("rgb", "rgba"):
            raise ValidationError("Convert HSV/Lab to RGB or grayscale before SimpleITK conversion")
    anchor = document.get("root_spatial", {})
    spatial = payload.get("spatial", {})
    root_size = [source["size_xyz"][i] for i in axes]
    check_spatial({"spatial": anchor}, root_size)
    if not np.allclose(anchor["to_root"], np.eye(3), rtol=0, atol=1e-9):
        raise ValidationError("Bridge anchor must use the original medical plane grid")
    check_spatial(payload, [array.shape[1], array.shape[0]])
    if not spatial.get("root_id"):
        if not assume_same_grid or [array.shape[1], array.shape[0]] != root_size:
            raise ValidationError(
                "Image spatial provenance is missing; explicitly allow same-grid "
                "reattachment only for unchanged dimensions"
            )
        transform = np.eye(3)
    else:
        if spatial.get("root_id") != anchor.get("root_id") or spatial.get("root_size") != root_size:
            raise ValidationError("The OpenCV image belongs to a different medical plane")
        transform = np.asarray(spatial.get("to_root"), dtype=float)
    if (
        transform.shape != (3, 3)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[2], [0, 0, 1], rtol=0, atol=1e-9)
    ):
        raise ValidationError(
            "Projective image geometry cannot be represented by SimpleITK spacing"
        )
    basis = np.asarray(source["direction_xyz"]).reshape(n, n) @ np.diag(source["spacing_xyz"])
    origin = np.asarray(source["origin_xyz"]) + basis[:, axes] @ transform[:2, 2]
    if n == 3:
        origin += basis[:, axis] * index
    output_basis = basis.copy()
    output_basis[:, axes] = basis[:, axes] @ transform[:2, :2]
    spacing = np.linalg.norm(output_basis, axis=0)
    if not np.all(spacing > 0):
        raise ValidationError("Medical image geometry is degenerate")
    direction = output_basis / spacing
    if not np.allclose(direction.T @ direction, np.eye(n), rtol=0, atol=1e-6):
        raise ValidationError(
            "Sheared OpenCV geometry cannot be represented by orthonormal "
            "SimpleITK direction cosines; explicitly resample onto a physical grid"
        )
    if n == 3:
        array = np.expand_dims(array, axis=2 - axis)
    output_size = list(reversed(array.shape[:n]))
    check_allocation(output_size, 1 if image.ndim == 2 else image.shape[2], array.dtype.itemsize)
    native = sitk.GetImageFromArray(np.ascontiguousarray(array), isVector=image.ndim == 3)
    updated = {
        **source,
        "size_xyz": list(native.GetSize()),
        "origin_xyz": origin.tolist(),
        "spacing_xyz": spacing.tolist(),
        "direction_xyz": direction.reshape(-1).tolist(),
    }
    set_information(native, updated)
    payload["simpleitk"] = document
    return MedicalImage(native, payload)


def same_geometry(a, b, *, tolerance=1e-6):
    return a.GetSize() == b.GetSize() and all(
        np.allclose(getattr(a, name)(), getattr(b, name)(), rtol=0, atol=tolerance)
        for name in ("GetOrigin", "GetSpacing", "GetDirection")
    )
