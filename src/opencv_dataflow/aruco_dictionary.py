"""Portable custom ArUco dictionaries with rotation-aware qualification."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from .calibration_revisions import digest
from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

DICTIONARIES = {
    name.removeprefix("DICT_").lower(): getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_") and isinstance(getattr(cv2.aruco, name), int)
}
DICTIONARY_CHOICES = (*DICTIONARIES, "custom")
ARUCO_DICTIONARY_INPUT = Port("dictionary", "data", required=False, required_when_connected=True)
MAX_MARKERS = 1024
_MAX_GENERATED_MARKERS = 256
_MAX_FILE_BYTES = 128 * 1024
_MODEL = "aruco_dictionary_v1"


@dataclass(frozen=True)
class Qualification:
    inter_id: int | None
    self_rotation: int
    native_bytes: bytes

    @property
    def minimum(self):
        return (
            min(self.inter_id, self.self_rotation)
            if self.inter_id is not None
            else self.self_rotation
        )

    @property
    def maximum_safe_bits(self):
        return max(0, (self.minimum - 1) // 2)

    def report(self):
        return {
            "minimum_hamming_distance": self.minimum,
            "minimum_inter_id_distance": self.inter_id,
            "minimum_self_rotation_distance": self.self_rotation,
            "maximum_safe_correction_bits": self.maximum_safe_bits,
        }


@lru_cache(maxsize=16)
def _qualify(size, codewords):
    # Each persisted row is the canonical orientation. The native byte layout
    # is rebuilt by OpenCV rather than serialized as an implementation detail.
    bits = (np.frombuffer("".join(codewords).encode("ascii"), np.uint8) - ord("0")).reshape(
        -1, size, size
    )
    rotations = np.stack([np.rot90(bits, i, axes=(1, 2)) for i in range(4)], axis=1)
    octets = np.packbits(rotations.reshape(len(bits), 4, -1), axis=2)
    words = np.pad(octets, ((0, 0), (0, 0), (0, 8 - octets.shape[2]))).view(np.uint64)[..., 0]
    self_distance = int(np.bitwise_count(words[:, :1] ^ words[:, 1:]).min())
    between = min(
        (
            int(np.bitwise_count(word ^ words[i + 1 :]).min())
            for i, word in enumerate(words[:-1, 0])
        ),
        default=None,
    )
    packed = np.concatenate(
        [cv2.aruco.Dictionary.getByteListFromBits(np.ascontiguousarray(row)) for row in bits]
    )
    return Qualification(between, self_distance, packed.tobytes())


def _inspect(record):
    if (
        not isinstance(record, dict)
        or record.get("model") != _MODEL
        or set(record)
        != {
            "model",
            "marker_size",
            "codewords",
            "max_correction_bits",
            "generation",
            "sha256",
        }
    ):
        raise ValidationError("Connect an ArUco dictionary record", "dictionary")
    size, words = record["marker_size"], record["codewords"]
    if (
        type(size) is not int
        or not 4 <= size <= 8
        or not isinstance(words, list)
        or not 1 <= len(words) <= MAX_MARKERS
        or any(
            not isinstance(row, str) or len(row) != size**2 or set(row) - {"0", "1"}
            for row in words
        )
    ):
        raise ValidationError("Invalid ArUco marker size or canonical codewords", "dictionary")
    created = record["generation"]
    if (
        not isinstance(created, dict)
        or set(created) != {"seed", "opencv_version", "inherited_count"}
        or type(created["seed"]) is not int
        or not 0 <= created["seed"] < 2**31
        or not isinstance(created["opencv_version"], str)
        or not 1 <= len(created["opencv_version"]) <= 64
        or type(created["inherited_count"]) is not int
        or not 0 <= created["inherited_count"] <= len(words)
    ):
        raise ValidationError("Invalid ArUco generation metadata", "dictionary")
    correction = record["max_correction_bits"]
    if type(correction) is not int or not 0 <= correction <= (size**2 - 1) // 2:
        raise ValidationError("Invalid ArUco correction budget", "dictionary")
    if record["sha256"] != digest({key: value for key, value in record.items() if key != "sha256"}):
        raise ValidationError(
            "ArUco dictionary checksum differs from the saved codewords", "dictionary"
        )
    qualified = _qualify(size, tuple(words))
    if qualified.minimum == 0 or correction > qualified.maximum_safe_bits:
        raise ValidationError(
            "ArUco codewords overlap at the declared rotational correction limit", "dictionary"
        )
    return qualified


def _report(record, qualified):
    return {
        "marker_size": record["marker_size"],
        "marker_count": len(record["codewords"]),
        "max_correction_bits": record["max_correction_bits"],
        "sha256": record["sha256"],
        **record["generation"],
        **qualified.report(),
    }


def compile_dictionary(record):
    qualified = _inspect(record)
    size = record["marker_size"]
    octets = (size * size + 7) // 8
    packed = np.frombuffer(qualified.native_bytes, np.uint8).copy().reshape(-1, octets, 4)
    return cv2.aruco.Dictionary(packed, size, record["max_correction_bits"])


def select_dictionary(name, record=None):
    if name == "custom":
        return compile_dictionary(record)
    if record is not None:
        raise ValidationError("Select Custom to use the connected ArUco dictionary", "dictionary")
    return cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])


def _codewords(native):
    return tuple(
        "".join(str(bit) for bit in (native.getMarkerBits(i) != 0).astype(np.uint8).ravel())
        for i in range(len(native.bytesList))
    )


class CreateArucoDictionary(Block):
    type_id, title = "calibration.create_aruco_dictionary", "Create custom ArUco dictionary"
    category = block_type = "Calibration"
    subtype = "Grid and coordinates"
    inputs = (Port("base_dictionary", "data", required=False, required_when_connected=True),)
    outputs = (Port("dictionary", "data", required=False), Port("report", "data"))
    defaults = {
        "marker_size": 6,
        "marker_count": 24,
        "seed": 0,
        "base_family": "none",
        "minimum_hamming_distance": 0,
        "max_correction_bits": -1,
    }
    parameter_choices = {"base_family": ("none", *DICTIONARIES)}
    parameter_ranges = {
        "marker_size": NumericRange(4, 8, whole=True),
        "marker_count": NumericRange(1, _MAX_GENERATED_MARKERS, whole=True),
        "seed": NumericRange(0, 2**31 - 1, whole=True),
        "minimum_hamming_distance": NumericRange(0, 64, whole=True),
        "max_correction_bits": NumericRange(-1, 31, whole=True),
    }
    parameter_help = {
        "marker_size": "Data modules per side. The printed black border is separate.",
        "marker_count": "Number of ID rows to retain or generate.",
        "seed": "OpenCV random seed for the new codewords.",
        "base_family": "Optional predefined prefix. Retained IDs keep their original bit patterns.",
        "minimum_hamming_distance": "Minimum separation between distinct ID and rotation patterns.",
        "max_correction_bits": "Maximum corrected modules. Use -1 for the qualified maximum.",
    }
    input_help = {"base_dictionary": "Optional custom dictionary prefix in place of Base family."}
    output_help = {
        "dictionary": "Qualified canonical codewords in ID order, with a checksum.",
        "report": "Distances between IDs and rotations, and the qualified correction budget.",
    }
    parallel_safe = process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        size, count, seed = (int(p[key]) for key in ("marker_size", "marker_count", "seed"))
        base = inputs.get("base_dictionary")
        if base is not None:
            if p["base_family"] != "none":
                raise ValidationError(
                    "Choose a connected base or a predefined base family", "base_family"
                )
            base = compile_dictionary(base)
        elif p["base_family"] != "none":
            base = select_dictionary(p["base_family"])
        if base is not None and base.markerSize != size:
            raise ValidationError("Base dictionary bit size must match Marker size", "marker_size")
        inherited = min(count, len(base.bytesList)) if base is not None else 0
        if self._cancelled():
            return Result(
                {"dictionary": NO_VALUE, "report": {}},
                passed=False,
                fail_reasons=["aruco_dictionary_cancelled"],
            )
        native = cv2.aruco.extendDictionary(count, size, base, seed)
        if (
            native.markerSize != size
            or len(native.bytesList) != count
            or inherited
            and not np.array_equal(native.bytesList[:inherited], base.bytesList[:inherited])
        ):
            raise ValueError("OpenCV changed the dictionary dimensions or inherited IDs")
        codewords = _codewords(native)
        qualified = _qualify(size, codewords)
        requested = int(p["max_correction_bits"])
        correction = qualified.maximum_safe_bits if requested == -1 else requested
        record = {
            "model": _MODEL,
            "marker_size": size,
            "codewords": list(codewords),
            "max_correction_bits": correction,
            "generation": {
                "seed": seed,
                "opencv_version": cv2.__version__,
                "inherited_count": inherited,
            },
        }
        record["sha256"] = digest(record)
        report = _report(record, qualified)
        report["native_correction_bits"] = int(native.maxCorrectionBits)
        reasons = []
        if qualified.minimum < max(1, int(p["minimum_hamming_distance"])):
            reasons.append("aruco_dictionary_separation")
        if correction > qualified.maximum_safe_bits:
            reasons.append("aruco_dictionary_correction_budget")
        if self._cancelled():
            reasons.append("aruco_dictionary_cancelled")
        return Result(
            {"dictionary": NO_VALUE if reasons else record, "report": report},
            passed=not reasons,
            fail_reasons=reasons,
            message=f"{count} IDs, minimum Hamming distance {qualified.minimum}, "
            f"{qualified.maximum_safe_bits} safely correctable bits",
        )


def read_dictionary_file(path):
    with path.open("rb") as stream:
        raw = stream.read(_MAX_FILE_BYTES + 1)
    if len(raw) > _MAX_FILE_BYTES:
        raise ValidationError("ArUco dictionary file exceeds 128 KiB", "path")
    record = json.loads(raw)
    return record, _inspect(record)


def write_dictionary_file(path, record):
    _inspect(record)
    contents = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if path.exists():
        current, _ = read_dictionary_file(path)
        if current != record:
            raise FileExistsError("Choose a new path to preserve the existing ArUco dictionary")
        return
    stream = NamedTemporaryFile(prefix=".aruco-", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(contents)
        try:
            if os.name == "nt":
                temporary.rename(path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            current, _ = read_dictionary_file(path)
            if current != record:
                raise
    finally:
        temporary.unlink(missing_ok=True)


class _DictionaryFile(Block):
    defaults = {"path": "", "path_base": "project"}
    parameter_choices = {"path_base": ("project", "working_directory")}
    parameter_help = {
        "path": "Portable ArUco dictionary JSON file.",
        "path_base": "Base for relative file paths.",
    }
    project_path_parameters = ("path",)
    subtype = "Calibration profiles"

    def _path(self):
        path = self.parameters["path"]
        if not isinstance(path, str) or not path.strip():
            raise ValidationError("Choose an ArUco dictionary file", "path")
        path = Path(path)
        if self.parameters["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError(
                    "Save the project before using a relative dictionary path", "path"
                )
            return self.project_dir / path
        return path


class LoadArucoDictionary(_DictionaryFile):
    type_id, title = "input.aruco_dictionary", "Load ArUco dictionary"
    category = block_type = "Input"
    outputs = (Port("dictionary", "data"), Port("report", "data"))
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = self._path()
        record, qualified = read_dictionary_file(path)
        return Result(
            {"dictionary": record, "report": _report(record, qualified)},
            message=f"Loaded {len(record['codewords'])} ArUco IDs",
        )


class SaveArucoDictionary(_DictionaryFile):
    type_id, title = "output.aruco_dictionary", "Save ArUco dictionary"
    category = block_type = "Output"
    inputs = (Port("dictionary", "data"),)
    process_inputs_readonly = True

    def process(self, inputs):
        path = self._path()
        write_dictionary_file(path, inputs["dictionary"])
        return Result(message=f"Saved the dictionary to {path.name}")
