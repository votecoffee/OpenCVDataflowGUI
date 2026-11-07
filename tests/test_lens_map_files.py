"""Exercise lens map round trips, archive validation, and safe publication."""

import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import packet_for

_PROFILE = {
    "model": "opencv_pinhole_v1",
    "camera_matrix": [[45, 0, 31.5], [0, 46, 23.5], [0, 0, 1]],
    "distortion": [-0.22, 0.015, 0.001, 0, 0],
    "image_size": [64, 48],
    "rms_pixels": 0.04,
    "units": "mm",
}
_PLANES = ("map_x", "map_y")


def _maps(registry):
    block = registry.create(
        "calibration.lens_maps", parameters={"camera_matrix_mode": "optimal", "alpha": 0.4}
    )
    maps = block.execute({"calibration": _PROFILE})
    assert maps is not None, block.error
    return maps


def _packets(result):
    return {name: packet_for(result, name) for name in _PLANES}


def _save(registry, path, maps):
    block = registry.create("output.lens_maps", parameters={"path": str(path)})
    assert block.execute(_packets(maps)) is not None, block.error
    return block


def test_round_trip_preserves_planes_profile_roi_and_corrected_coordinates(tmp_path):
    registry = builtin_registry()
    original = _maps(registry)
    path = tmp_path / "maps.npz"
    _save(registry, path, original)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"metadata", *_PLANES}

    loader = registry.create(
        "input.lens_maps", parameters={"path": "maps.npz", "path_base": "project"}
    )
    loader.project_dir = tmp_path
    restored = loader.execute({})
    assert restored is not None, loader.error
    for name in _PLANES:
        np.testing.assert_array_equal(restored.outputs[name], original.outputs[name])
        for key in ("spatial", "lens_remap_map"):
            assert restored.payloads[name][key] == original.payloads[name][key]
    assert restored.outputs["rectified_calibration"] == original.outputs["rectified_calibration"]
    assert restored.outputs["valid_roi"] == original.outputs["valid_roi"]

    image = np.random.default_rng(81).integers(0, 256, (48, 64), dtype=np.uint8)
    remap = registry.create("cv.remap")
    first = remap.execute({"image": image, **_packets(original)})
    second = remap.execute({"image": image, **_packets(restored)})
    assert first is not None and second is not None, remap.error
    np.testing.assert_array_equal(first.outputs["image"], second.outputs["image"])
    assert first.payloads["image"]["spatial"] == second.payloads["image"]["spatial"]


def test_saver_preserves_existing_file_and_rejects_reversed_pair(tmp_path):
    registry = builtin_registry()
    maps = _maps(registry)
    path = tmp_path / "maps.npz"
    saver = _save(registry, path, maps)
    before = path.read_bytes()
    assert saver.execute(_packets(maps)) is None
    assert "Existing lens map files" in saver.error
    assert path.read_bytes() == before

    saver.parameters["path"] = str(tmp_path / "reversed.npz")
    inputs = _packets(maps)
    assert saver.execute({"map_x": inputs["map_y"], "map_y": inputs["map_x"]}) is None
    assert "matching X and Y" in saver.error
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("competing_writer", (False, True))
def test_saver_discards_partial_temp_file_and_preserves_a_concurrent_writer(
    tmp_path, monkeypatch, competing_writer
):
    registry = builtin_registry()
    maps = _maps(registry)
    path = tmp_path / "maps.npz"
    saver = registry.create("output.lens_maps", parameters={"path": str(path)})
    native_savez = np.savez

    def interrupted(stream, **arrays):
        if competing_writer:
            native_savez(stream, **arrays)
            path.write_bytes(b"other writer")
        else:
            stream.write(b"partial archive")
            raise OSError("write failed")

    monkeypatch.setattr(np, "savez", interrupted)
    assert saver.execute(_packets(maps)) is None
    if competing_writer:
        assert path.read_bytes() == b"other writer"
        assert list(tmp_path.iterdir()) == [path]
    else:
        assert not list(tmp_path.iterdir())


def test_loader_rejects_modified_values_even_with_valid_zip_checksums(tmp_path):
    registry = builtin_registry()
    path = tmp_path / "maps.npz"
    _save(registry, path, _maps(registry))
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["map_x"][0, 0] += 1
    np.savez(path, **arrays)
    loader = registry.create("input.lens_maps", parameters={"path": str(path)})
    assert loader.execute({}) is None
    assert "checksum" in loader.error


def test_loader_checks_array_header_shape_before_reading_its_body(tmp_path):
    registry = builtin_registry()
    good = tmp_path / "maps.npz"
    bad = tmp_path / "bad.npz"
    _save(registry, good, _maps(registry))
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f4", "fortran_order": False, "shape": (2**30, 2**30)}
    )
    with ZipFile(good) as source, ZipFile(bad, "w") as destination:
        for name in source.namelist():
            data = header.getvalue() if name == "map_x.npy" else source.read(name)
            destination.writestr(name, data)
    loader = registry.create("input.lens_maps", parameters={"path": str(bad)})
    assert loader.execute({}) is None
    assert "Unexpected shape or type" in loader.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_archive_example_corrects_the_distorted_grid(profile):
    project = Path(__file__).parents[1] / "examples/custom tools/lens_map_files/lens_map_files.json"
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result.outputs["image"]
    corrected = graph.blocks["remap"].last_result.outputs["image"]
    assert corrected.shape == source.shape == (144, 192)
    assert np.mean(corrected[22:122, 24]) < np.mean(source[22:122, 24]) - 50
