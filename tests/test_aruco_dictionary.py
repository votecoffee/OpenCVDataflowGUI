"""Stored ID order, rotational separation and native bit correction."""

import json
from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest

from opencv_dataflow.aruco_dictionary import compile_dictionary, read_dictionary_file
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_revisions import digest
from opencv_dataflow.charuco import observation_board
from opencv_dataflow.coded_rig_observations import _layout
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/custom_aruco_dictionary"
CREATE = "calibration.create_aruco_dictionary"
DETECT = "inspect.aruco_markers"
GENERATE_TARGET = "calibration.generate_fiducial_target"


@pytest.fixture(scope="module")
def fixture():
    return run_path(str(EXAMPLE / "build_example.py"))


@pytest.fixture(scope="module")
def record(fixture):
    block = builtin_registry().create(CREATE, parameters=fixture["SETTINGS"])
    result = block.execute({})
    assert result and result.passed, block.error
    return result.outputs["dictionary"]


def test_seeded_words_and_rotational_distance_are_independently_reproducible(fixture, record):
    from_asset, qualified = read_dictionary_file(EXAMPLE / "assets/dictionary.json")
    assert from_asset == record
    # Every unique rotation has a disjoint six-error Hamming ball here.
    # Compare canonical rows and each rotation without native packed bytes.
    rot_codes = [
        [np.rot90(np.fromiter(map(int, word), dtype=np.uint8).reshape(6, 6), k) for k in range(4)]
        for word in record["codewords"]
    ]
    across = min(
        np.count_nonzero(old[0] != other)
        for i, old in enumerate(rot_codes)
        for row in rot_codes[i + 1 :]
        for other in row
    )
    within = min(np.count_nonzero(row[0] != code) for row in rot_codes for code in row[1:])
    assert (across, within, qualified.minimum, qualified.maximum_safe_bits) == (13, 14, 13, 6)
    native = compile_dictionary(record)
    for marker_id, bits in enumerate(rot_codes):
        for rotated in bits:
            assert native.getDistanceToId(rotated, marker_id, True) == 0

    altered = builtin_registry().create(CREATE, parameters={**fixture["SETTINGS"], "seed": 539})
    changed = altered.execute({})
    assert changed and changed.passed
    assert changed.outputs["dictionary"]["codewords"] != record["codewords"]


def test_extensions_retain_ids_from_predefined_and_custom_prefixes(record):
    registry = builtin_registry()
    predefined = registry.create(
        CREATE, parameters={"base_family": "4x4_50", "marker_count": 55, "marker_size": 4}
    )
    result = predefined.execute({})
    assert result and result.passed, predefined.error
    assert result.outputs["report"]["inherited_count"] == 50
    expected = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    compiled = compile_dictionary(result.outputs["dictionary"])
    np.testing.assert_array_equal(compiled.bytesList[:50], expected.bytesList)

    custom = registry.create(CREATE, parameters={"marker_count": 27, "marker_size": 6})
    result = custom.execute({"base_dictionary": record})
    assert result and result.passed, custom.error
    assert result.outputs["dictionary"]["codewords"][:24] == record["codewords"]
    assert result.outputs["report"]["inherited_count"] == 24


def test_budget_and_rotational_collision_are_rejected(fixture, record):
    registry = builtin_registry()
    bad = registry.create(CREATE, parameters={**fixture["SETTINGS"], "max_correction_bits": 7})
    result = bad.execute({})
    assert result and not result.passed
    assert result.outputs["dictionary"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "aruco_dictionary_correction_budget"

    body = {name: value for name, value in record.items() if name != "sha256"}
    rotating_copy = "".join(
        np.rot90(np.array(list(record["codewords"][0])).reshape(6, 6)).reshape(-1)
    )
    collision = {**body, "codewords": [*body["codewords"][:-1], rotating_copy]}
    collision["sha256"] = digest(collision)
    with pytest.raises(ValidationError, match="overlap"):
        compile_dictionary(collision)


def test_saved_dictionaries_are_idempotent_and_reject_tampered_files(tmp_path, record):
    path = tmp_path / "fixture.json"
    registry = builtin_registry()
    output = registry.create("output.aruco_dictionary", parameters={"path": str(path)})
    loader = registry.create("input.aruco_dictionary", parameters={"path": str(path)})
    assert output.execute({"dictionary": record}).passed
    before = path.read_bytes()
    assert output.execute({"dictionary": record}).passed
    assert path.read_bytes() == before
    assert loader.execute({}).outputs["dictionary"] == record

    changed = {k: v for k, v in record.items() if k != "sha256"}
    changed["max_correction_bits"] = 1
    changed["sha256"] = digest(changed)
    assert output.execute({"dictionary": changed}) is None
    assert path.read_bytes() == before
    assert "preserve the existing" in output.error

    tampered = json.loads(before)
    tampered["max_correction_bits"] = 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert loader.execute({}) is None
    assert "checksum differs" in loader.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_renders_loads_corrects_and_preserves_id_order(fixture, profile):
    graph = Graph.load(EXAMPLE / "custom_aruco_dictionary.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        generate, target, loaded, grid, correct = (
            graph.blocks[name].last_result
            for name in ("generate", "target", "load", "grid", "correct")
        )
        assert all(result and result.passed for result in (generate, target, loaded, grid, correct))
        assert generate.outputs["dictionary"] == loaded.outputs["dictionary"]
        spec = target.outputs["target"]
        assert spec["marker_ids"] == fixture["ORDER"]
        assert spec["dictionary_record"] == loaded.outputs["dictionary"]
        dictionary, units, ids, _ = _layout(spec)
        assert dictionary.markerSize == 6 and units == "mm" and ids == fixture["ORDER"]
        by_id = {row["id"]: row["center"] for row in grid.outputs["markers"]}
        assert grid.outputs["points"] == [by_id[row] for row in fixture["ORDER"]]
        assert [m["id"] for m in correct.outputs["markers"]] == [fixture["DAMAGED_ID"]]
    finally:
        graph.close()


@pytest.mark.parametrize("flips,rate,passes", [(2, 1.0, True), (2, 0.5, False), (3, 1.0, False)])
def test_bit_error_bound_applies_to_rotated_marker_images(fixture, record, flips, rate, passes):
    native = compile_dictionary(record)
    image = np.rot90(fixture["damaged_marker"](native, flips=flips)).copy()
    tool = builtin_registry().create(
        DETECT,
        parameters={
            "dictionary": "custom",
            "required_ids": [fixture["DAMAGED_ID"]],
            "error_correction_rate": rate,
        },
    )
    result = tool.execute({"image": image, "dictionary": record})
    assert result and result.passed == passes, tool.error
    assert result.outputs["count"] == int(passes)
    if not passes:
        assert result.outputs["points"] is NO_VALUE


def test_custom_charuco_geometry_replays_and_3d_pose_uses_same_ids(fixture, record):
    registry = builtin_registry()
    settings = {
        "dictionary": "custom",
        "columns": 5,
        "rows": 5,
        "units": "mm",
        "marker_length": 20.0,
        "square_length": 30.0,
    }
    create = registry.create(GENERATE_TARGET, parameters={**settings, "dpi": 127})
    generated = create.execute({"dictionary": record})
    assert generated and generated.passed, create.error
    detect = registry.create("calibration.charuco", parameters=settings)
    detected = detect.execute({"dictionary": record, "image": generated.outputs["image"]})
    assert detected and detected.passed, detect.error
    assert detected.outputs["corner_count"] == 16
    spec, _, board, _ = observation_board(detected.outputs["observation"])
    assert spec["dictionary_record"] == record
    assert board.getDictionary().markerSize == 6

    image = fixture["damaged_marker"](compile_dictionary(record), flips=0)
    width, height = image.shape[::-1]
    camera = camera_record([[550, 0, 88], [0, 550, 93], [0, 0, 1]], [], [width, height])
    pose = registry.create("3d.aruco_pose", parameters={"dictionary": "custom"})
    result = pose.execute({"dictionary": record, "image": image, "camera": camera})
    assert result and result.passed, pose.error
    assert result.outputs["pose"]["marker_id"] == fixture["DAMAGED_ID"]
