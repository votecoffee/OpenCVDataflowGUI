"""Bounded CPU hierarchical feature-selection segmentation."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool
from .region_proposals import label_boundary_image, segmentation_image
from .selection import SELECTION_INPUTS, selection_mask

_MAX_PIXELS = 2_100_000
_MAX_SEEDS = 4096


class HfsSegments(ImageTool):
    type_id, title = "cv.hfs_segments", "Hierarchical feature-selection segmentation"
    subtype = "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"), Port("labels"), Port("boundaries", visual_type="mask"),
        Port("count", "integer"),
    )
    defaults = {
        "stage1_threshold": 0.08,
        "stage1_min_size": 100,
        "stage2_threshold": 0.28,
        "stage2_min_size": 200,
        "spatial_weight": 0.6,
        "superpixel_size": 8,
    }
    parameter_ranges = {
        "stage1_threshold": NumericRange(0, 1000),
        "stage1_min_size": NumericRange(1, _MAX_PIXELS, whole=True),
        "stage2_threshold": NumericRange(0, 1000),
        "stage2_min_size": NumericRange(1, _MAX_PIXELS, whole=True),
        "spatial_weight": NumericRange(0, 1000),
        "superpixel_size": NumericRange(8, 4096, whole=True),
    }
    parameter_help = {
        "stage1_threshold": "First graph merge threshold, applied to the SLIC superpixels.",
        "stage1_min_size": "First graph pass merges adjacent regions smaller than this pixel area.",
        "stage2_threshold": "Second graph merge threshold, applied to the first pass regions.",
        "stage2_min_size": "Second graph pass merges adjacent regions smaller than this area.",
        "spatial_weight": "SLIC proximity weight. Larger weights favor compact regions.",
        "superpixel_size": "Initial SLIC grid spacing in pixels. Increase it for larger frames.",
    }
    input_help = {
        "image": "Grayscale, BGR or RGB uint8 image. The CPU runs full-frame with up to 2.1 MP.",
        "mask": "Optional output selection, intersected with the supplied region.",
        "region": "Optional output selection in the image coordinate frame.",
    }
    output_help = {
        "image": "Selected pixels are filled with their full-frame region's mean color.",
        "labels": "Native positive region IDs as int32. Zero marks unselected pixels.",
        "boundaries": "Two-sided 4-neighbor boundaries, masked to the selected area.",
        "count": "Number of positive region IDs present in the selected label image.",
    }
    payload_reference = "image"
    process_inputs_readonly = True

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        p = self.parameters
        spacing = int(p["superpixel_size"])
        if min(height, width) < spacing:
            raise ValidationError("Superpixel size must fit both image axes", "superpixel_size")
        if height * width > _MAX_PIXELS:
            raise ValidationError("Crop the image to at most 2.1 million pixels for HFS")
        seeds = ((height + spacing - 1) // spacing) * ((width + spacing - 1) // spacing)
        if seeds > _MAX_SEEDS:
            raise ValidationError(
                f"Increase superpixel size or crop the image to fit {_MAX_SEEDS:,} SLIC seeds",
                "superpixel_size",
            )
        image = segmentation_image(source)
        selection = selection_mask(inputs, source)
        hfs_class = getattr(getattr(cv2, "hfs", None), "HfsSegment", None)
        factory = getattr(hfs_class, "create", None)
        if factory is None:
            raise ValidationError("HFS requires OpenCV contrib with the hfs Python module")
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        segmenter = factory(
            height, width, float(p["stage1_threshold"]), int(p["stage1_min_size"]),
            float(p["stage2_threshold"]), int(p["stage2_min_size"]),
            float(p["spatial_weight"]), spacing,
        )
        native = segmenter.performSegmentCpu(bgr, False)
        if (
            native is None or native.shape != (height, width) or native.dtype != np.uint16
            or native.min() == 0 or native.max() > seeds
        ):
            raise ValidationError("OpenCV returned an invalid HFS label plane")
        labels = native.astype(np.int32)
        ids = labels.ravel()
        areas = np.bincount(ids)
        areas[areas == 0] = 1
        original = np.asarray(source)
        pixels = original.reshape(ids.size, -1)
        mean_lookup = np.stack(
            [np.bincount(ids, weights=channel) / areas for channel in pixels.T], axis=1
        )
        output = np.rint(mean_lookup[ids]).astype(np.uint8).reshape(original.shape)
        if selection is not None:
            outside = selection == 0
            output[outside] = original[outside]
            labels[outside] = 0
        count = int(np.count_nonzero(np.bincount(labels.ravel())[1:]))
        return Result(
            {
                "image": ImageArray(output, image_format(source)),
                "labels": labels,
                "boundaries": label_boundary_image(labels),
                "count": count,
            },
            message=f"{count} hierarchical segments in the output selection",
        )
