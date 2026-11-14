"""Independent pixel/physical-coordinate checks for the optional medical bridge."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph, Port, validate_value
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, metadata
from opencv_dataflow.simpleitk_image import (
    MedicalImage,
    from_opencv,
    image_metadata,
    same_geometry,
    to_opencv,
    validate_metadata,
)

sitk = pytest.importorskip("SimpleITK")


def native(shape=(7, 9, 11), dtype=np.int16):
    image = sitk.GetImageFromArray(np.arange(np.prod(shape), dtype=dtype).reshape(shape))
    image.SetSpacing((0.5, 0.8, 2.0))
    image.SetOrigin((13.0, -7.0, 40.0))
    angle = 0.37
    image.SetDirection((np.cos(angle), 0, np.sin(angle), 0, 1, 0, -np.sin(angle), 0, np.cos(angle)))
    image.SetMetaData("example:description", "Public software test fixture")
    return image


def run(kind, inputs, parameters=None):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute(inputs)
    assert block.state == "pass", block.error
    return block


@pytest.mark.parametrize(
    "axis,index,shape", [("x", 3, (7, 9)), ("y", 4, (7, 11)), ("z", 2, (9, 11))]
)
def test_oblique_anisotropic_slice_pixels_and_physical_points(axis, index, shape):
    original = native()
    wrapped = MedicalImage(original)
    pixels, bridge, _ = to_opencv(wrapped, axis=axis, index=index)
    assert pixels.shape == shape and pixels.dtype == np.int16
    output = from_opencv(pixels).image
    axes = bridge["plane_axes"]
    for x, y in ((0, 0), (3, 4), (shape[1] - 1, shape[0] - 1)):
        source_index = [0, 0, 0]
        source_index["xyz".index(axis)] = index
        source_index[axes[0]], source_index[axes[1]] = x, y
        output_index = source_index.copy()
        output_index["xyz".index(axis)] = 0
        assert output[output_index] == original[source_index]
        np.testing.assert_allclose(
            output.TransformIndexToPhysicalPoint(output_index),
            original.TransformIndexToPhysicalPoint(source_index),
            atol=1e-12,
        )
    assert output.GetMetaData("example:description") == original.GetMetaData("example:description")


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_two_dimensional_roundtrip_keeps_real_sample_type_and_geometry(dtype):
    array = np.arange(48).reshape(6, 8).astype(dtype)
    original = sitk.GetImageFromArray(array)
    original.SetSpacing([0.4, 1.7])
    original.SetOrigin([3, -9])
    original.SetDirection([0, -1, 1, 0])
    image, _, _ = to_opencv(MedicalImage(original))
    reconstructed = from_opencv(image).image
    np.testing.assert_array_equal(sitk.GetArrayFromImage(reconstructed), array)
    assert same_geometry(original, reconstructed)


def test_crop_resize_composes_pixel_center_mapping_into_physical_geometry():
    original = native()
    converted = run("sitk.to_opencv", {"image": MedicalImage(original)}, {"slice_index": 2})
    crop = run(
        "cv.crop",
        {"image": converted.last_result.outputs["image"]},
        {"x": 2, "y": 1, "width": 6, "height": 6},
    )
    resize = run(
        "cv.resize", {"image": crop.last_result.outputs["image"]}, {"width": 12, "height": 18}
    )
    final = run("sitk.from_opencv", {"image": resize.last_result.outputs["image"]})
    image = final.last_result.outputs["image"].image
    assert image.GetSize() == (12, 18, 1)
    for x, y in ((0, 0), (4, 7), (11, 17)):
        # OpenCV resize maps pixel centers by (out+0.5)/scale - 0.5.
        point = original.TransformContinuousIndexToPhysicalPoint(
            [2 + (x + 0.5) / 2 - 0.5, 1 + (y + 0.5) / 3 - 0.5, 2]
        )
        np.testing.assert_allclose(
            image.TransformIndexToPhysicalPoint([x, y, 0]), point, atol=1e-12
        )


def test_explicit_metadata_reattachment_requires_a_declared_same_grid_assumption():
    image, bridge, _ = to_opencv(MedicalImage(native()), index=2)
    plain = np.asarray(image).copy()
    with pytest.raises(ValueError, match="provenance is missing"):
        from_opencv(plain, bridge)
    recovered = from_opencv(plain, bridge, assume_same_grid=True)
    assert recovered.image.GetMetaData("example:description")
    with pytest.raises(ValueError, match="unchanged dimensions"):
        from_opencv(plain[:-1], bridge, assume_same_grid=True)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 0, 0], [0, 1, 0], [0.01, 0, 1]],
        [[1, 0.3, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_unrepresentable_projective_or_sheared_geometry_rejects(matrix):
    image, _, _ = to_opencv(MedicalImage(native()))
    image.metadata["spatial"]["to_root"] = matrix
    with pytest.raises(ValueError, match="Projective|Sheared"):
        from_opencv(image)


def test_different_plane_metadata_and_incorrect_grid_are_rejected():
    original = MedicalImage(native())
    image, _, _ = to_opencv(original, index=2)
    _, other, _ = to_opencv(original, index=3)
    with pytest.raises(ValueError, match="differs"):
        from_opencv(image, other)
    image.metadata["spatial"]["root_id"] = "another-plane"
    with pytest.raises(ValueError, match="different medical plane"):
        from_opencv(image)


def test_explicit_rgb_and_one_component_vectors_do_not_swap_or_lose_channels():
    rgb = np.zeros((5, 7, 3), np.uint16)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 100, 200, 300
    original = MedicalImage(sitk.GetImageFromArray(rgb, isVector=True))
    with pytest.raises(ValueError, match="explicitly"):
        to_opencv(original)
    pixels, _, _ = to_opencv(original, component_mode="rgb")
    assert pixels.color_format == "rgb"
    bgr = ImageArray(np.asarray(pixels)[..., ::-1].copy(), "bgr", pixels.metadata)
    restored = from_opencv(bgr)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(restored.image), rgb)
    one = MedicalImage(sitk.GetImageFromArray(np.full((5, 7, 1), 4, np.uint8), isVector=True))
    scalar, _, _ = to_opencv(one, component_mode="component")
    assert scalar.shape == (5, 7) and np.all(scalar == 4)


def test_insert_slice_preserves_other_voxels_and_rejects_transformed_or_foreign_slices():
    original = MedicalImage(native(dtype=np.float32))
    converted = run("sitk.to_opencv", {"image": original}, {"slice_index": 2})
    pixels = converted.last_result.outputs["image"].copy()
    pixels[:] = -1
    restored = run("sitk.from_opencv", {"image": pixels})
    inserted = run(
        "sitk.insert_slice", {"image": original, "slice": restored.last_result.outputs["image"]}
    )
    actual = sitk.GetArrayFromImage(inserted.last_result.outputs["image"].image)
    expected = sitk.GetArrayFromImage(original.image)
    expected[2] = -1
    np.testing.assert_array_equal(actual, expected)
    other = MedicalImage(original.image)
    failed = builtin_registry().create("sitk.insert_slice")
    failed.execute({"image": other, "slice": restored.last_result.outputs["image"]})
    assert failed.state == "error" and "reference volume" in failed.error


@pytest.mark.parametrize("suffix", [".mha", ".nrrd", ".nii.gz"])
def test_medical_file_roundtrip_preserves_samples_and_physical_grid_and_existing_files(
    tmp_path, suffix
):
    original = MedicalImage(native())
    path = tmp_path / f"medical{suffix}"
    original.image.SetMetaData("StudyNote", "Public software test fixture")
    writer = run("output.simpleitk", {"image": original}, {"path": str(path)})
    reader = run("input.simpleitk", {}, {"path": str(path)})
    restored = reader.last_result.outputs["image"].image
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(restored), sitk.GetArrayFromImage(original.image)
    )
    assert same_geometry(restored, original.image, tolerance=2e-6)
    if suffix != ".nii.gz":
        assert restored.GetMetaData("StudyNote") == "Public software test fixture"
    before = path.read_bytes()
    writer.execute({"image": original})
    assert writer.state == "error" and path.read_bytes() == before
    assert not list(tmp_path.glob(".simpleitk-*"))


def test_metadata_merge_and_filters_retain_dictionary_without_mutating_input():
    original = MedicalImage(native(dtype=np.float32))
    updated = run(
        "sitk.apply_metadata",
        {"image": original},
        {"tag_updates": {"note": "new"}, "remove_tags": ["example:description"]},
    )
    changed = updated.last_result.outputs["image"]
    assert changed.image.GetMetaData("note") == "new"
    assert not changed.image.HasMetaDataKey("example:description")
    assert original.image.HasMetaDataKey("example:description")
    for kind in ("sitk.window", "sitk.gaussian", "sitk.threshold", "sitk.otsu"):
        block = run(kind, {"image": changed})
        output = block.last_result.outputs["image"].image
        assert same_geometry(output, original.image)
        assert output.GetMetaData("note") == "new"


def test_label_metrics_have_independent_truth_and_inspection_failure():
    a = np.zeros((4, 5, 6), np.uint8)
    b = a.copy()
    a[:, :, :2], b[:, :, 1:3] = 1, 1
    prediction, truth = (
        MedicalImage(sitk.GetImageFromArray(a)),
        MedicalImage(sitk.GetImageFromArray(b)),
    )
    block = builtin_registry().create("sitk.label_overlap", parameters={"minimum_dice": 0.75})
    block.execute({"image": prediction, "reference": truth})
    assert block.state == "fail"
    metrics = block.last_result.outputs["metrics"]
    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["jaccard"] == pytest.approx(1 / 3)
    truth.image.SetOrigin([0, 0, 2])
    block.execute({"image": prediction, "reference": truth})
    assert block.state == "error" and "physical voxel grid" in block.error


def test_native_copies_and_typed_ports_isolate_buffers_and_fail_closed():
    original = MedicalImage(native())
    copied = copy.deepcopy(original)
    copied.image[0, 0, 0] = 99
    assert original.image[0, 0, 0] == 0
    sitk.GetArrayFromImage(copied.image)
    assert copied.image.GetMetaData("example:description") == "Public software test fixture"
    assert original.image.GetMetaData("example:description") == "Public software test fixture"
    assert copied.identity == original.identity
    with pytest.raises(TypeError, match="converter"):
        validate_value(Port("image", "sitk"), np.zeros((5, 6), np.uint8))
    registry = builtin_registry()
    graph = Graph()
    source, target = (
        graph.add(registry.create("input.simpleitk")),
        graph.add(registry.create("cv.gray")),
    )
    with pytest.raises(ValueError, match="Cannot connect"):
        graph.connect(source.id, "image", target.id, "image")


@pytest.mark.parametrize(
    "field,value",
    [
        ("spacing_xyz", [1, 0, 1]),
        ("direction_xyz", [1] * 9),
        ("origin_xyz", [float("nan"), 0, 0]),
        ("tags", {"wrong": 4}),
    ],
)
def test_malformed_metadata_rejects(field, value):
    document = image_metadata(native())
    document[field] = value
    with pytest.raises(ValueError):
        validate_metadata(document)


def test_tickets_and_pyramid_context_survive_native_roundtrip():
    from opencv_dataflow.tickets import TicketClock

    ticket = TicketClock().issue("test", "external", "medical-001")
    source = MedicalImage(native())
    converted = run("sitk.to_opencv", {"image": ValuePacket(source, {"ticket": ticket})})
    original = converted.last_result.outputs["image"]
    down = run("cv.pyramid_down", {"image": original})
    result = run("sitk.from_opencv", {"image": down.last_result.outputs["image"]})
    assert result.last_result.payloads["image"]["ticket"] == ticket
    assert result.last_result.outputs["image"].image.GetSpacing()[:2] == pytest.approx((1, 1.6))
    assert metadata(original)["simpleitk"]["model"] == "simpleitk_bridge_v1"


def test_missing_optional_dependency_leaves_catalog_and_other_groups_available():
    code = """
import sys, importlib.abc
class Missing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'SimpleITK': raise ModuleNotFoundError('SimpleITK intentionally absent')
sys.meta_path.insert(0, Missing())
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.toolbox import block_entry
r=builtin_registry()
for cls in r.types.values():
    if cls.__module__.endswith('.simpleitk_blocks'):
        assert block_entry(cls).path[0] == 'SimpleITK'
        r.create(cls.type_id).validate_parameters()
assert block_entry(r.types['skimage.multi_otsu']).path[0] == 'scikit-image'
assert block_entry(r.types['inspect.simple_classify_image']).path[0] == 'Simple Plugins'
b=r.create('input.simpleitk', parameters={'path':'missing.nii.gz'}); b.execute({})
assert b.state == 'error' and 'simpleitk' in b.error
assert 'SimpleITK' not in sys.modules
print('Optional dependency isolated')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_project_json_xml_and_toolbox_group_roundtrip(tmp_path):
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml
    from opencv_dataflow.toolbox import block_entry

    registry = builtin_registry()
    items = []
    for kind, cls in registry.types.items():
        if cls.__module__.endswith(".simpleitk_blocks"):
            block = registry.create(kind, block_id=kind)
            assert block_entry(cls).path[0] == "SimpleITK"
            items.append(block.to_project_item())
    assert len(items) == 12
    data = {"schema_version": 1, "blocks": items, "connections": []}
    restored = Graph.from_dict(json.loads(json.dumps(data)), registry)
    assert restored.to_dict() == data
    path = tmp_path / "medical.xml"
    export_project_xml(data, path)
    assert read_project_xml(path) == data


def test_otsu_mask_and_nondefault_controls_match_native():
    image = native(dtype=np.float32)
    mask = sitk.Cast(image > 100, sitk.sitkUInt8)
    block = run(
        "sitk.otsu",
        {"image": MedicalImage(image), "mask": MedicalImage(mask)},
        {
            "bins": 32,
            "inside": 9,
            "outside": 7,
            "return_bin_midpoint": False,
            "mask_value": 1,
            "mask_output": True,
        },
    )
    operator = sitk.OtsuThresholdImageFilter()
    operator.SetNumberOfHistogramBins(32)
    operator.SetInsideValue(9)
    operator.SetOutsideValue(7)
    operator.SetReturnBinMidpoint(False)
    operator.SetMaskValue(1)
    operator.SetMaskOutput(True)
    expected = operator.Execute(image, mask)
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(block.last_result.outputs["image"].image),
        sitk.GetArrayFromImage(expected),
    )
    assert block.last_result.outputs["threshold"] == operator.GetThreshold()


def test_float_window_range_and_absent_label_report_invalid_data():
    original = MedicalImage(native())
    block = builtin_registry().create("sitk.window", parameters={"output_max": 1e100})
    block.execute({"image": original})
    assert block.error_kind == "invalid"
    labels = MedicalImage(sitk.Image([4, 5, 6], sitk.sitkUInt8))
    block = builtin_registry().create("sitk.label_overlap")
    block.execute({"image": labels, "reference": labels})
    assert block.error_kind == "invalid" and "absent in both" in block.error


def test_medical_and_existing_tool_groups_are_distinct_in_editor(app):
    from PySide6.QtCore import Qt

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        roots = {
            window.library.topLevelItem(i).text(0)
            for i in range(window.library.topLevelItemCount())
        }
        assert {"SimpleITK", "scikit-image", "Simple Plugins"} <= roots
        matches = window.library.findItems(
            "SimpleITK to OpenCV", Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive, 0
        )
        assert len(matches) == 1 and matches[0].parent().parent().text(0) == "SimpleITK"
        window.add_block(matches[0])
        assert window.inspector.block.type_id == "sitk.to_opencv"
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
