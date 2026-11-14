"""Build nine medical interop projects and independent physical/label expectations."""

import json
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

from opencv_dataflow.assistant_api import ProjectSession

HERE = Path(__file__).resolve().parent
CASES = ("001", "003", "004")


def source(kind, case, labels=False):
    folder = "labels" if labels else "images"
    return {
        "op": "add",
        "type": "input.simpleitk",
        "id": kind,
        "parameters": {"path": f"../assets/msd_hippocampus/{folder}/hippocampus_{case}.nii.gz"},
    }


def add(kind, block_id, **parameters):
    return {"op": "add", "type": kind, "id": block_id, "parameters": parameters}


def wire(a, b, output="image", input="image"):
    return {"op": "connect", "source": a, "output": output, "target": b, "input": input}


def build():
    expected = {
        name: {} for name in ("metadata_roundtrip", "geometry_roundtrip", "label_roundtrip")
    }
    for case in CASES:
        path = HERE / f"assets/msd_hippocampus/images/hippocampus_{case}.nii.gz"
        original = sitk.ReadImage(str(path))
        labels = sitk.ReadImage(
            str(HERE / f"assets/msd_hippocampus/labels/hippocampus_{case}.nii.gz")
        )
        pixels, truth = sitk.GetArrayFromImage(original), sitk.GetArrayFromImage(labels)
        index = int(np.argmax(np.count_nonzero(truth, axis=(1, 2))))
        low, high = map(float, np.percentile(pixels, [1, 99]))
        w, h, _ = original.GetSize()
        common = [
            source("mri", case),
            add("layout.comment", "attribution"),
            {
                "op": "update",
                "id": "mri",
                "fields": {"ticket_settings": {"mode": "fixed", "value": f"hippocampus_{case}"}},
            },
            {
                "op": "update",
                "id": "attribution",
                "fields": {
                    "name": "Medical Segmentation Decathlon — CC BY-SA 4.0",
                    "description": "Task04 Hippocampus, Vanderbilt University Medical Center; "
                    "Medical Segmentation Decathlon contributors, Simpson et al. "
                    "Original MRI and expert labels, subset only. "
                    "https://registry.opendata.aws/msd/ — see assets attribution. "
                    "Interop demonstration; label roundtrip is not an MRI "
                    "segmentation model or a clinical performance estimate.",
                },
            },
        ]
        projects = {
            "metadata_roundtrip": [
                add("sitk.read_metadata", "original_metadata"),
                add("sitk.window", "window", lower=low, upper=high, output_type="uint8"),
                add("sitk.to_opencv", "plane", slice_index=index),
                add("cv.gaussian_blur", "blur", kernel=3, sigma=0.8),
                add("sitk.from_opencv", "restore"),
                add(
                    "sitk.apply_metadata",
                    "tags",
                    tag_updates={"ProcessingNote": "Explicit window then OpenCV Gaussian blur"},
                ),
                add("sitk.read_metadata", "inspect_metadata"),
                add("sitk.to_opencv", "review"),
                wire("mri", "original_metadata"),
                wire("mri", "window"),
                wire("window", "plane"),
                wire("plane", "blur"),
                wire("blur", "restore"),
                wire("plane", "restore", "metadata", "metadata"),
                wire("restore", "tags"),
                wire("original_metadata", "tags", "metadata", "metadata"),
                wire("tags", "inspect_metadata"),
                wire("inspect_metadata", "review"),
            ],
            "geometry_roundtrip": [
                add("sitk.to_opencv", "plane", slice_index=index),
                add("cv.crop", "crop", x=2, y=3, width=w - 4, height=h - 6),
                add("cv.resize", "resize", width=2 * (w - 4), height=2 * (h - 6)),
                add("sitk.from_opencv", "restore"),
                add("sitk.read_metadata", "inspect_metadata"),
                add("sitk.to_opencv", "review"),
                add("cv.colorize_scalar", "display", colormap="bone"),
                wire("mri", "plane"),
                wire("plane", "crop"),
                wire("crop", "resize"),
                wire("resize", "restore"),
                wire("plane", "restore", "metadata", "metadata"),
                wire("restore", "inspect_metadata"),
                wire("inspect_metadata", "review"),
                wire("review", "display"),
            ],
            "label_roundtrip": [
                source("labels", case, labels=True),
                add("sitk.to_opencv", "plane", slice_index=index),
                add("cv.median_blur", "median", kernel=3),
                add("sitk.from_opencv", "restore"),
                add("sitk.insert_slice", "insert"),
                add("sitk.label_overlap", "overlap", label=1, minimum_dice=0.9),
                add(
                    "cv.colorize_scalar",
                    "label_display",
                    range_mode="fixed",
                    minimum=0.0,
                    maximum=2.0,
                    colormap="viridis",
                ),
                add("sitk.window", "mri_window", lower=low, upper=high, output_type="uint8"),
                add("sitk.to_opencv", "mri_plane", slice_index=index),
                wire("labels", "plane"),
                wire("plane", "median"),
                wire("median", "restore"),
                wire("labels", "insert"),
                wire("restore", "insert", input="slice"),
                wire("insert", "overlap"),
                wire("labels", "overlap", input="reference"),
                wire("median", "label_display"),
                wire("mri", "mri_window"),
                wire("mri_window", "mri_plane"),
            ],
        }
        for folder, steps in projects.items():
            directory = HERE / folder
            directory.mkdir(exist_ok=True)
            with ProjectSession(directory=directory) as session:
                session.apply([*common, *steps, {"op": "arrange"}])
                data = session.graph.to_dict()
                (directory / f"hippocampus_{case}.json").write_bytes(
                    (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
                )
        expected["metadata_roundtrip"][case] = {
            "source_size_xyz": list(original.GetSize()),
            "slice_index": index,
            "plane_size_xyz": [w, h, 1],
            "plane_origin_xyz": list(original.TransformIndexToPhysicalPoint([0, 0, index])),
            "processing_note": "Explicit window then OpenCV Gaussian blur",
        }
        expected["geometry_roundtrip"][case] = {
            "slice_index": index,
            "size_xyz": [2 * (w - 4), 2 * (h - 6), 1],
            "spacing_xyz": [0.5, 0.5, 1.0],
            "origin_xyz": list(
                original.TransformContinuousIndexToPhysicalPoint([1.75, 2.75, index])
            ),
        }
        modified = truth.copy()
        modified[index] = cv2.medianBlur(truth[index], 3)
        a, b = modified == 1, truth == 1
        intersection = int(np.count_nonzero(a & b))
        count_a, count_b = int(np.count_nonzero(a)), int(np.count_nonzero(b))
        expected["label_roundtrip"][case] = {
            "slice_index": index,
            "label": 1,
            "predicted_voxels": count_a,
            "reference_voxels": count_b,
            "intersection_voxels": intersection,
            "dice": 2 * intersection / (count_a + count_b),
            "jaccard": intersection / int(np.count_nonzero(a | b)),
            "changed_voxels": int(np.count_nonzero(modified != truth)),
        }
    for folder, cases in expected.items():
        (HERE / folder / "expected.json").write_bytes(
            (json.dumps(cases, indent=2, allow_nan=False) + "\n").encode()
        )


if __name__ == "__main__":
    build()
