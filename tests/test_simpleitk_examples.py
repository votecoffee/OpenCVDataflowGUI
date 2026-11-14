"""Public MRI/label provenance and the saved medical interoperability projects."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

sitk = pytest.importorskip("SimpleITK")
BASE = Path(__file__).resolve().parents[1] / "examples/SimpleITK Tools"


@pytest.mark.parametrize("folder", ["metadata_roundtrip", "geometry_roundtrip", "label_roundtrip"])
@pytest.mark.parametrize("case", ["001", "003", "004"])
def test_saved_real_medical_project_matches_independent_expectations(folder, case):
    expected = json.loads((BASE / folder / "expected.json").read_text())[case]
    graph = Graph.load(BASE / folder / f"hippocampus_{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks()), {
            b.id: b.error for b in graph.processing_blocks() if b.state != "pass"
        }
        assert any(b.layout_only and "CC BY-SA" in b.name for b in graph.blocks.values())
        if folder == "label_roundtrip":
            result = graph.blocks["overlap"].last_result
            metrics = result.outputs["metrics"]
            for key in ("dice", "jaccard", "predicted_voxels", "reference_voxels"):
                assert metrics[key] == pytest.approx(expected[key])
            truth = sitk.GetArrayFromImage(
                graph.blocks["labels"].last_result.outputs["image"].image
            )
            actual = sitk.GetArrayFromImage(
                graph.blocks["insert"].last_result.outputs["image"].image
            )
            mask = np.ones(truth.shape, bool)
            mask[expected["slice_index"]] = False
            np.testing.assert_array_equal(actual[mask], truth[mask])
            assert np.count_nonzero(actual != truth) == expected["changed_voxels"]
            assert result.payloads["image"]["ticket"]["value"] is not None
        else:
            image = graph.blocks["restore"].last_result.outputs["image"].image
            origin = (
                expected["plane_origin_xyz"]
                if folder == "metadata_roundtrip"
                else expected["origin_xyz"]
            )
            np.testing.assert_allclose(image.GetOrigin(), origin, atol=1e-10)
            if folder == "geometry_roundtrip":
                assert list(image.GetSize()) == expected["size_xyz"]
                np.testing.assert_allclose(image.GetSpacing(), expected["spacing_xyz"], atol=1e-10)
            else:
                assert list(image.GetSize()) == expected["plane_size_xyz"]
                tags = graph.blocks["inspect_metadata"].last_result.outputs["metadata"]["tags"]
                assert tags["ProcessingNote"] == expected["processing_note"]
                original = graph.blocks["original_metadata"].last_result.outputs["metadata"]["tags"]
                assert all(tags[key] == value for key, value in original.items())
    finally:
        graph.close()


def test_public_source_files_have_exact_recorded_bytes_and_matching_expert_label_grids():
    root = BASE / "assets/msd_hippocampus"
    manifest = json.loads((root / "SOURCES.json").read_text())
    assert manifest["license"] == "CC-BY-SA-4.0" and len(manifest["files"]) == 6
    for record in manifest["files"]:
        raw = (root / record["path"]).read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    for case in ("001", "003", "004"):
        name = f"hippocampus_{case}.nii.gz"
        image, labels = (
            sitk.ReadImage(str(root / "images" / name)),
            sitk.ReadImage(str(root / "labels" / name)),
        )
        assert image.GetSize() == labels.GetSize()
        for getter in ("GetSpacing", "GetOrigin", "GetDirection"):
            assert getattr(image, getter)() == getattr(labels, getter)()
        assert set(np.unique(sitk.GetArrayFromImage(labels))) == {0, 1, 2}


def test_medical_example_opens_and_previews_in_editor(app):
    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        graph = Graph.load(BASE / "metadata_roundtrip/hippocampus_001.json", window.registry)
        graph.run()
        window.replace_graph(graph)
        window.scene.nodes["restore"].setSelected(True)
        assert window.inspector.can_preview()
        assert "assume_same_grid" in window.inspector.fields
        assert graph.blocks["review"].last_output_images["image"].ndim == 2
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
