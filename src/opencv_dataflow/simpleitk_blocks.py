"""Medical I/O, physical metadata, OpenCV bridges and a focused SimpleITK baseline."""

import copy
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .simpleitk_image import (
    MAX_BYTES,
    MedicalImage,
    check_allocation,
    from_opencv,
    geometry,
    image_metadata,
    pixel_types,
    require_medical,
    same_geometry,
    set_information,
    sitk_module,
    to_opencv,
    validate_metadata,
)


class MedicalTool(Block):
    category, block_type, subtype = "SimpleITK", "Image processing", "Processing"
    inputs = (Port("image", "sitk"),)
    outputs = (Port("image", "sitk"), Port("info", "data"))
    payload_reference = "image"
    process_inputs_readonly = parallel_safe = True

    def wrapped_result(self, image, **kwargs):
        output = MedicalImage(image, copy.deepcopy(self.input_payloads.get("image", {})))
        return Result({"image": output, "info": output.describe()}, **kwargs)


def file_path(block):
    path = Path(block.parameters["path"])
    if not block.parameters["path"]:
        raise ValidationError("Choose a medical image path", "path")
    if not path.is_absolute() and block.parameters["path_base"] == "project":
        if block.project_dir is None:
            raise ValidationError("Save/open the project before using project-relative files")
        path = block.project_dir / path
    return path


class ReadMedicalImage(MedicalTool):
    type_id, title, block_type, subtype = "input.simpleitk", "Read medical image", "Input", "I/O"
    inputs = ()
    defaults = {"path": "", "path_base": "project"}
    parameter_choices = {"path_base": ("project", "working_directory")}
    parameter_help = {
        "path": "Single 2D/3D medical image, such as NIfTI, MetaImage or NRRD. "
        "DICOM series assembly is a separate planned tool."
    }
    preview_safe = False

    def process(self, inputs):
        sitk = sitk_module()
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(file_path(self)))
        reader.ReadImageInformation()
        dtype = pixel_types().get(reader.GetPixelID())
        if dtype is None:
            raise ValidationError("Only real scalar/vector medical images are supported")
        check_allocation(list(reader.GetSize()), reader.GetNumberOfComponents(), dtype.itemsize)
        return self.wrapped_result(reader.Execute())


class WriteMedicalImage(MedicalTool):
    type_id, title, block_type, subtype = "output.simpleitk", "Write medical image", "Output", "I/O"
    defaults = {"path": "", "path_base": "project", "compression": True}
    parameter_choices = {"path_base": ("project", "working_directory")}
    parameter_help = {
        "path": "New single-file .nii/.nii.gz/.mha/.nrrd destination; existing "
        "files are preserved. Physical geometry is written by SimpleITK."
    }
    preview_safe = parallel_safe = False

    def process(self, inputs):
        sitk = sitk_module()
        image = require_medical(inputs["image"])
        path = file_path(self)
        suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix.lower()
        if suffix not in (".nii", ".nii.gz", ".mha", ".nrrd"):
            raise ValidationError(
                "Use a single-file .nii, .nii.gz, .mha or .nrrd destination", "path"
            )
        if path.exists():
            raise FileExistsError(f"Destination already exists: {path}")
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=path.parent, prefix=".simpleitk-", suffix=suffix, delete=False
            ) as stream:
                temporary = Path(stream.name)
            sitk.WriteImage(image, str(temporary), bool(self.parameters["compression"]))
            if temporary.stat().st_size > MAX_BYTES:
                raise ValidationError("Encoded medical image exceeded 256 MiB")
            os.link(temporary, path)  # Atomic exclusive publication; never replace another file.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return Result({"image": inputs["image"], "info": {"path": str(path), "written": True}})


class MedicalToOpenCV(MedicalTool):
    type_id, title, subtype = "sitk.to_opencv", "SimpleITK to OpenCV", "Converters"
    outputs = (Port("image"), Port("metadata", "data"), Port("info", "data"))
    defaults = {"axis": "z", "slice_index": 0, "component_mode": "scalar_only", "component": 0}
    parameter_choices = {
        "axis": ("x", "y", "z"),
        "component_mode": ("scalar_only", "rgb", "rgba", "component"),
    }
    parameter_ranges = {
        "slice_index": NumericRange(0, 65535, whole=True),
        "component": NumericRange(0, 15, whole=True),
    }
    parameter_help = {
        "axis": "3D slice normal in native XYZ index space; both remaining axes keep their order.",
        "slice_index": "Zero-based source slice; 2D images require zero.",
        "component_mode": "Scalar samples retain depth/units. Color vectors require explicit "
        "RGB/RGBA interpretation; arbitrary vectors require component selection.",
    }

    def process(self, inputs):
        supplied = inputs["image"]
        value = MedicalImage(
            supplied.image, copy.deepcopy(self.input_payloads["image"]), supplied.identity
        )
        p = self.parameters
        image, bridge, payload = to_opencv(
            value,
            axis=p["axis"],
            index=int(p["slice_index"]),
            component_mode=p["component_mode"],
            component=int(p["component"]),
        )
        return Result(
            {
                "image": image,
                "metadata": bridge,
                "info": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "slice_axis": bridge["slice_axis"],
                    "slice_index": bridge["slice_index"],
                },
            },
            payloads={"image": payload, "metadata": {}, "info": {}},
            message="Raw samples; physical geometry and tags attached to image payload",
        )


class OpenCVToMedical(MedicalTool):
    type_id, title, subtype = "sitk.from_opencv", "OpenCV to SimpleITK", "Converters"
    inputs = (Port("image"), Port("metadata", "data", required=False, required_when_connected=True))
    defaults = {"assume_same_grid": False}
    parameter_help = {
        "assume_same_grid": "Explicitly reattach metadata when spatial provenance "
        "is absent and dimensions are unchanged. Leave off when "
        "crop/resize/warp provenance is available."
    }

    def process(self, inputs):
        output = from_opencv(
            inputs["image"],
            inputs.get("metadata"),
            assume_same_grid=self.parameters["assume_same_grid"],
        )
        return Result(
            {"image": output, "info": output.describe()},
            payloads={"image": output.metadata, "info": {}},
            message="Physical geometry recomputed from pixel-center mapping; tags copied",
        )


class ReadMedicalMetadata(MedicalTool):
    type_id, title, subtype = "sitk.read_metadata", "Read medical metadata", "Metadata"
    outputs = (Port("image", "sitk"), Port("metadata", "data"))

    def process(self, inputs):
        return Result(
            {
                "image": inputs["image"],
                "metadata": image_metadata(require_medical(inputs["image"])),
            },
            payloads={"metadata": {}},
        )


class ApplyMedicalMetadata(MedicalTool):
    type_id, title, subtype = "sitk.apply_metadata", "Apply medical metadata", "Metadata"
    inputs = (
        Port("image", "sitk"),
        Port("metadata", "data", required=False, required_when_connected=True),
    )
    defaults = {"tag_updates": {}, "remove_tags": [], "apply_geometry": False}
    parameter_help = {
        "apply_geometry": "Explicitly copy geometry from metadata input; actual "
        "image dimensions must match. Does not resample pixels.",
        "tag_updates": "String key/value dictionary; overrides copied tags.",
        "remove_tags": "Tag keys to erase after merging metadata.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        tags, removed = p["tag_updates"], p["remove_tags"]
        probe = {
            "model": "simpleitk_metadata_v1",
            "dimension": 2,
            "size_xyz": [1, 1],
            "origin_xyz": [0, 0],
            "spacing_xyz": [1, 1],
            "direction_xyz": [1, 0, 0, 1],
            "tags": tags,
        }
        validate_metadata(probe)
        if not isinstance(removed, list) or any(not isinstance(k, str) for k in removed):
            raise ValidationError("Remove tags must be a list of string keys", "remove_tags")

    def process(self, inputs):
        native = require_medical(inputs["image"])
        output = sitk_module().Image(native)
        document = copy.deepcopy(inputs.get("metadata", image_metadata(native)))
        validate_metadata(document)
        document["tags"].update(self.parameters["tag_updates"])
        set_information(output, document, copy_geometry=self.parameters["apply_geometry"])
        for key in self.parameters["remove_tags"]:
            if output.HasMetaDataKey(key):
                output.EraseMetaData(key)
        image_metadata(output)  # Validate the merged dictionary's total size.
        return self.wrapped_result(output)


class InsertMedicalSlice(MedicalTool):
    type_id, title, subtype = "sitk.insert_slice", "Reinsert processed medical slice", "Converters"
    inputs = (Port("image", "sitk"), Port("slice", "sitk"))

    def process(self, inputs):
        sitk = sitk_module()
        reference, plane = require_medical(inputs["image"]), require_medical(inputs["slice"])
        bridge = self.input_payloads["slice"].get("simpleitk", {})
        source = bridge.get("image", {})
        if (
            reference.GetDimension() != 3
            or bridge.get("model") != "simpleitk_bridge_v1"
            or bridge.get("source_id") != inputs["image"].identity
            or source.get("dimension") != 3
            or bridge.get("selected_component") is not None
        ):
            raise ValidationError("Slice must come from this reference volume's converter")
        if any(source.get(key) != value for key, value in geometry(reference).items()):
            raise ValidationError("Reference volume geometry differs from the slice's source")
        axis, index = bridge["slice_axis"], bridge["slice_index"]
        size, start = list(reference.GetSize()), [0] * 3
        size[axis], start[axis] = 1, index
        expected = sitk.Extract(reference, size, start)
        if not same_geometry(expected, plane) or expected.GetPixelID() != plane.GetPixelID():
            raise ValidationError(
                "Reinsertion requires a full unchanged slice grid and pixel type; "
                "cropped/resized slices need explicit resampling"
            )
        result = sitk.Paste(reference, plane, size, [0, 0, 0], start)
        set_information(result, image_metadata(reference))
        return self.wrapped_result(result, message=f"Replaced axis {axis} slice {index}")


class MedicalWindow(MedicalTool):
    type_id, title = "sitk.window", "Medical intensity window"
    defaults = {
        "lower": 0.0,
        "upper": 1000.0,
        "output_min": 0.0,
        "output_max": 255.0,
        "output_type": "float32",
    }
    parameter_choices = {"output_type": ("float32", "float64", "uint8", "uint16")}
    parameter_ranges = {
        key: NumericRange() for key in ("lower", "upper", "output_min", "output_max")
    }
    parameter_help = {
        "lower": "Raw sample lower limit; clipped values map to output_min.",
        "upper": "Raw sample upper limit; clipped values map to output_max.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["lower"] >= p["upper"] or p["output_min"] >= p["output_max"]:
            raise ValidationError(
                "Window and output bounds must increase",
                "lower",
                "upper",
                "output_min",
                "output_max",
            )
        if p["output_type"] in ("uint8", "uint16"):
            maximum = np.iinfo(p["output_type"]).max
            if p["output_min"] < 0 or p["output_max"] > maximum:
                raise ValidationError(
                    "Output range exceeds selected unsigned type",
                    "output_min",
                    "output_max",
                    "output_type",
                )
        elif max(abs(p["output_min"]), abs(p["output_max"])) > float(
            np.finfo(p["output_type"]).max
        ):
            raise ValidationError("Output range exceeds selected floating type", "output_type")

    def process(self, inputs):
        sitk, image = sitk_module(), require_medical(inputs["image"])
        p = self.parameters
        if image.GetNumberOfComponentsPerPixel() != 1:
            raise ValidationError("Intensity window expects a scalar image")
        pixel_id = {
            "float32": sitk.sitkFloat32,
            "float64": sitk.sitkFloat64,
            "uint8": sitk.sitkUInt8,
            "uint16": sitk.sitkUInt16,
        }[p["output_type"]]
        output = sitk.IntensityWindowing(
            sitk.Cast(image, sitk.sitkFloat64),
            p["lower"],
            p["upper"],
            p["output_min"],
            p["output_max"],
        )
        output = sitk.Cast(output, pixel_id)
        set_information(output, image_metadata(image))
        return self.wrapped_result(output)


class MedicalGaussian(MedicalTool):
    type_id, title = "sitk.gaussian", "Physical Gaussian smoothing"
    defaults = {"sigma": 1.0, "normalize_across_scale": False}
    parameter_ranges = {"sigma": NumericRange(0.000001, 10000)}
    parameter_help = {
        "sigma": "Gaussian standard deviation in the image's physical units; "
        "not pixels. SimpleITK does not infer the unit name."
    }

    def process(self, inputs):
        image = require_medical(inputs["image"])
        if min(image.GetSize()) < 4 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValidationError("Recursive Gaussian needs scalar images with >=4 voxels per axis")
        output = sitk_module().SmoothingRecursiveGaussian(
            image, float(self.parameters["sigma"]), bool(self.parameters["normalize_across_scale"])
        )
        set_information(output, image_metadata(image))
        return self.wrapped_result(output)


class MedicalThreshold(MedicalTool):
    type_id, title, subtype = "sitk.threshold", "Medical binary threshold", "Segmentation"
    defaults = {"lower": 0.0, "upper": 1000.0, "inside": 1, "outside": 0}
    parameter_ranges = {
        "lower": NumericRange(),
        "upper": NumericRange(),
        "inside": NumericRange(0, 255, whole=True),
        "outside": NumericRange(0, 255, whole=True),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["lower"] > p["upper"]:
            raise ValidationError("Lower threshold must not exceed upper", "lower", "upper")

    def process(self, inputs):
        image = require_medical(inputs["image"])
        p = self.parameters
        if image.GetNumberOfComponentsPerPixel() != 1:
            raise ValidationError("Threshold expects a scalar image")
        # Floating work preserves noninteger threshold bounds for integer inputs.
        sitk = sitk_module()
        output = sitk.BinaryThreshold(
            sitk.Cast(image, sitk.sitkFloat64),
            p["lower"],
            p["upper"],
            int(p["inside"]),
            int(p["outside"]),
        )
        set_information(output, image_metadata(image))
        return self.wrapped_result(output)


class MedicalOtsu(MedicalTool):
    type_id, title, subtype = "sitk.otsu", "Medical Otsu threshold", "Segmentation"
    outputs = (*MedicalTool.outputs, Port("threshold", "number"))
    inputs = (
        Port("image", "sitk"),
        Port("mask", "sitk", required=False, required_when_connected=True),
    )
    defaults = {
        "bins": 128,
        "inside": 0,
        "outside": 1,
        "return_bin_midpoint": True,
        "mask_value": 1,
        "mask_output": True,
    }
    parameter_ranges = {
        "bins": NumericRange(2, 65536, whole=True),
        "inside": NumericRange(0, 255, whole=True),
        "outside": NumericRange(0, 255, whole=True),
        "mask_value": NumericRange(0, 255, whole=True),
    }
    parameter_help = {
        "mask_value": "Only this uint8 mask label contributes to the histogram.",
        "mask_output": "When a mask is supplied, zero output outside its selection.",
    }

    def process(self, inputs):
        image = require_medical(inputs["image"])
        if image.GetNumberOfComponentsPerPixel() != 1:
            raise ValidationError("Otsu threshold expects a scalar image")
        operator = sitk_module().OtsuThresholdImageFilter()
        operator.SetNumberOfHistogramBins(int(self.parameters["bins"]))
        operator.SetInsideValue(int(self.parameters["inside"]))
        operator.SetOutsideValue(int(self.parameters["outside"]))
        operator.SetReturnBinMidpoint(bool(self.parameters["return_bin_midpoint"]))
        if "mask" in inputs:
            mask = require_medical(inputs["mask"])
            if mask.GetPixelID() != sitk_module().sitkUInt8 or not same_geometry(image, mask):
                raise ValidationError(
                    "Otsu mask must be scalar uint8 on the same physical voxel grid"
                )
            operator.SetMaskValue(int(self.parameters["mask_value"]))
            operator.SetMaskOutput(bool(self.parameters["mask_output"]))
            output = operator.Execute(image, mask)
        else:
            output = operator.Execute(image)
        set_information(output, image_metadata(image))
        result = self.wrapped_result(output)
        result.outputs["threshold"] = float(operator.GetThreshold())
        return result


class MedicalLabelOverlap(MedicalTool):
    type_id, title, subtype = "sitk.label_overlap", "Medical label overlap / Dice", "Measurements"
    inputs = (Port("image", "sitk"), Port("reference", "sitk"))
    outputs = (*MedicalTool.outputs, Port("metrics", "data"))
    defaults = {
        "label": 1,
        "minimum_dice": 0.0,
        "geometry_tolerance": 0.000001,
        "allow_integral_float": True,
    }
    parameter_ranges = {
        "label": NumericRange(1, 65535, whole=True),
        "minimum_dice": NumericRange(0, 1),
        "geometry_tolerance": NumericRange(0, 0.001),
    }
    parameter_help = {
        "label": "Nonzero label measured against reference; background excluded.",
        "geometry_tolerance": "Absolute tolerance for spacing/origin/direction. "
        "No implicit registration or resampling is performed.",
        "allow_integral_float": "Accept floating label files only when every voxel is finite "
        "and exactly a whole number. Probability maps are rejected.",
    }

    def process(self, inputs):
        sitk = sitk_module()
        image, reference = require_medical(inputs["image"]), require_medical(inputs["reference"])
        if image.GetDimension() != reference.GetDimension() or not same_geometry(
            image, reference, tolerance=float(self.parameters["geometry_tolerance"])
        ):
            raise ValidationError("Prediction/reference must share the same physical voxel grid")
        for value in (image, reference):
            kind = pixel_types()[value.GetPixelID()].kind
            if value.GetNumberOfComponentsPerPixel() != 1:
                raise ValidationError("Label overlap needs scalar label images")
            if kind == "f" and self.parameters["allow_integral_float"]:
                flat = sitk.GetArrayViewFromImage(value).reshape(-1)
                for start in range(0, flat.size, 1_000_000):
                    chunk = flat[start : start + 1_000_000]
                    if not np.isfinite(chunk).all() or not np.equal(chunk, np.rint(chunk)).all():
                        raise ValidationError(
                            "Floating label images must contain finite whole "
                            "numbers; probability maps are not labels"
                        )
            elif kind not in "iu":
                raise ValidationError("Enable integral floating labels or supply integer labels")
        label = int(self.parameters["label"])
        # Binary masks bound the filter's label table even for high-valued uint64 labels.
        predicted, truth = sitk.Equal(image, label), sitk.Equal(reference, label)
        statistics = sitk.StatisticsImageFilter()
        statistics.Execute(predicted)
        predicted_count = int(statistics.GetSum())
        statistics.Execute(truth)
        reference_count = int(statistics.GetSum())
        if predicted_count == reference_count == 0:
            raise ValidationError("Selected label is absent in both inputs; Dice is undefined")
        # Treat explicitly tolerated geometry differences as rounding, without resampling.
        truth.CopyInformation(predicted)
        operator = sitk.LabelOverlapMeasuresImageFilter()
        operator.Execute(predicted, truth)
        dice, jaccard = (
            float(operator.GetDiceCoefficient()),
            float(operator.GetJaccardCoefficient()),
        )
        passed = dice >= float(self.parameters["minimum_dice"])
        metrics = {
            "label": label,
            "dice": dice,
            "jaccard": jaccard,
            "predicted_voxels": predicted_count,
            "reference_voxels": reference_count,
            "false_negative_error": (
                float(operator.GetFalseNegativeError()) if reference_count else None
            ),
            "false_positive_error": (
                float(operator.GetFalsePositiveError()) if predicted_count else None
            ),
        }
        return Result(
            {"image": inputs["image"], "info": inputs["image"].describe(), "metrics": metrics},
            passed=passed,
            score=dice,
            message=f"Label {label}: Dice {dice:.6f}, Jaccard {jaccard:.6f}",
            fail_reasons=[]
            if passed
            else [{"code": "label_dice", "message": "Label overlap below minimum Dice"}],
        )


SIMPLEITK_BLOCKS = (
    ReadMedicalImage,
    WriteMedicalImage,
    MedicalToOpenCV,
    OpenCVToMedical,
    ReadMedicalMetadata,
    ApplyMedicalMetadata,
    InsertMedicalSlice,
    MedicalWindow,
    MedicalGaussian,
    MedicalThreshold,
    MedicalOtsu,
    MedicalLabelOverlap,
)
