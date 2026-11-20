"""Replay one recorded point-cloud file per activation with exact capture provenance."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .cloud_io import read_cloud
from .core import Port, Result
from .geometry3d_common import cloud_graphics
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .replay import ReplaySource

MANIFEST_FORMAT = "opencv_dataflow_cloud_recording_v1"
_MAX_MANIFEST_BYTES = 8 * 1024**2
_MAX_SCANS = 20_000
_SCAN_EXTENSIONS = {".ply", ".pcd", ".obj"}


@dataclass(frozen=True)
class ScanRecord:
    path: str
    timestamp_ns: int
    frame: str
    units: str


@dataclass(frozen=True)
class CloudRecording:
    path: Path
    clock: str
    scans: tuple[ScanRecord, ...]


def _unique_fields(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field in point-cloud manifest")
    return result


def _label(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 128
        and value.strip()
        and all(ord(character) >= 32 for character in value)
    )


def _relative_scan_path(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(ord(character) < 32 for character in value)
        or ":" in value
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or PurePosixPath(value).suffix.lower() not in _SCAN_EXTENSIONS
    ):
        raise ValidationError("Scan paths must be relative PLY, PCD or OBJ files using '/'", "path")
    return value


def _read_manifest(path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_MANIFEST_BYTES + 1)
        if not 0 < len(raw) <= _MAX_MANIFEST_BYTES:
            raise ValueError("Manifest must be nonempty and at most 8 MiB")
        manifest = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_fields)
    except (OSError, ValueError, RecursionError) as exc:
        raise ValidationError(f"Could not read point-cloud manifest: {exc}", "path") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.keys() != {"format", "clock", "scans"}
        or manifest["format"] != MANIFEST_FORMAT
    ):
        raise ValidationError(f"Choose an {MANIFEST_FORMAT} manifest", "path")
    clock = manifest["clock"]
    if not _label(clock):
        raise ValidationError("Name the recording clock", "path")
    records = manifest["scans"]
    if not isinstance(records, list) or not 1 <= len(records) <= _MAX_SCANS:
        raise ValidationError("Recording requires 1..20,000 scans", "path")

    scans = []
    previous = -1
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict) or record.keys() != {
            "path",
            "timestamp_ns",
            "frame",
            "units",
        }:
            raise ValidationError(f"Scan {index} needs path, timestamp_ns, frame and units", "path")
        stamp, frame, units = record["timestamp_ns"], record["frame"], record["units"]
        if type(stamp) is not int or not previous < stamp <= 2**63 - 1:
            raise ValidationError(
                f"Scan {index} timestamps must be nonnegative, strictly increasing integers",
                "path",
            )
        if not _label(frame) or units not in ("m", "mm", "unitless"):
            raise ValidationError(f"Scan {index} needs a frame and m/mm/unitless units", "path")
        scans.append(ScanRecord(_relative_scan_path(record["path"]), stamp, frame, units))
        previous = stamp
    return CloudRecording(path, clock, tuple(scans))


class RecordedCloud(ReplaySource):
    type_id, title, category = "input.cloud_recording", "Recorded point-cloud sequence", "Input"
    block_type, subtype = "Input", "Point clouds"
    description = "Replay recorded PLY, PCD or OBJ scans in manifest order with capture timestamps."
    outputs = (Port("cloud", "cloud"), Port("info", "data"))
    flow_primary = ("cloud",)
    project_path_parameters = ("path",)
    defaults = {"path": "", "repeat": "stop", "max_points": 1_000_000, "track_source_rows": True}
    parameter_choices = {"repeat": ("stop", "loop")}
    parameter_ranges = {"max_points": NumericRange(1, 2_000_000, whole=True)}
    parameter_help = {
        "path": "Manifest JSON relative to the project. Scan paths are relative to the manifest.",
        "repeat": "Stop at the final scan or loop. Run All in the editor rewinds and stops.",
        "max_points": "Maximum declared point rows per file, including missing organized pixels.",
        "track_source_rows": "Add original file row indices when the scan has no source_index.",
    }
    output_help = {
        "cloud": "One scan with its stored units, frame, attributes and recorded_cloud payload.",
        "info": "Clock, exact capture timestamp, source path, scan index and point count.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._sequence = {"settings": None, "cursor": 0, "finished": False}
        self._recording = None
        self._pending = None

    def snapshot_runtime(self):
        return copy.deepcopy(self._sequence)

    def restore_runtime(self, snapshot):
        if snapshot["settings"] != self._sequence["settings"]:
            self._recording = None
        self._sequence = copy.deepcopy(snapshot)
        self._pending = None

    def copy_runtime_from(self, other):
        self.restore_runtime(other.snapshot_runtime())
        self._recording = other._recording  # Immutable manifest, no decoded clouds.

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["path"], str):
            raise ValidationError("Choose a point-cloud manifest path", "path")
        if "repeat" in self.recipe_bindings:
            raise ValidationError("Repeat is a run setting and cannot be recipe-bound", "repeat")

    @property
    def exhausted(self):
        settings = (self.project_dir, {**self.parameters, **self.last_recipe_parameters})
        return (
            self.finite_source
            and self._sequence["finished"]
            and (self._sequence["settings"] == settings)
        )

    def _load_recording(self):
        name = self.parameters["path"]
        path = Path(name)
        if not name.strip() or "\0" in name or path.suffix.lower() != ".json":
            raise ValidationError("Choose a point-cloud manifest JSON file", "path")
        if not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError(
                    "Save or open the project before using a relative path", "path"
                )
            path = self.project_dir / path
        return _read_manifest(path.resolve())

    def _load_scan(self, index):
        record = self._recording.scans[index]
        directory = self._recording.path.parent
        try:
            path = (directory / record.path).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Scan must stay inside the recording folder")
            if not path.is_file():
                raise ValueError("Scan file is missing or is not a regular file")
            cloud = read_cloud(
                path,
                record.units,
                record.frame,
                int(self.parameters["max_points"]),
                track_source_rows=self.parameters["track_source_rows"],
            )
            if (cloud.frame, cloud.units) != (record.frame, record.units):
                raise ValueError("Scan file frame or units differ from the manifest")
            if not len(cloud.points):
                raise ValueError("Scan contains no usable points")
        except (OSError, ValueError) as exc:
            raise ValidationError(f"Scan {index + 1} ({record.path}): {exc}", "path") from exc
        return cloud

    def issue_source_ticket(self):
        settings = (self.project_dir, dict(self.parameters))
        if self._sequence["settings"] != settings:
            self.reset_runtime()
            self._sequence["settings"] = settings
        state = self._sequence
        self._pending = None
        if state["finished"] or self._cancelled():
            return None
        try:
            if self._recording is None:
                self._recording = self._load_recording()
            index = state["cursor"]
            cloud = self._load_scan(index)
            if self._cancelled():
                return None
            ticket = super().issue_source_ticket()
            self._pending = index, cloud
            state["cursor"] += 1
            if state["cursor"] == len(self._recording.scans):
                if self.finite_source:
                    state["finished"] = True
                else:
                    state["cursor"] = 0
            return ticket
        except Exception:
            state["finished"] = True
            raise

    def process(self, inputs):
        if self._pending is None:
            message = (
                "Point-cloud recording paused"
                if self._cancelled()
                else ("Point-cloud recording complete")
            )
            return Result({"cloud": NO_VALUE, "info": NO_VALUE}, message=message)

        index, cloud = self._pending
        self._pending = None
        recording = self._recording
        scan = recording.scans[index]
        capture = {
            "clock": recording.clock,
            "timestamp_ns": scan.timestamp_ns,
            "scan_index": index,
            "path": scan.path,
            "frame": scan.frame,
            "units": scan.units,
        }
        cloud.metadata["recorded_cloud"] = capture
        info = {
            **capture,
            "manifest": recording.path.name,
            "scan_count": len(recording.scans),
            "point_count": len(cloud.points),
        }
        return Result(
            {"cloud": cloud, "info": info},
            payloads={"info": {"recorded_cloud": capture}},
            graphics=cloud_graphics(self, cloud),
            message=f"Scan {index + 1}/{len(recording.scans)} · {scan.path}",
        )
