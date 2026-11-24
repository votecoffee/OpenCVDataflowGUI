"""Private child entry point for OpenCV Edge Drawing."""

import json
import sys

import cv2
import numpy as np

MAX_PATH_POINTS = 262_144
MAX_PATHS = 4096

_NATIVE_FIELDS = {
    "parameter_free": "PFmode",
    "gradient_threshold": "GradientThresholdValue",
    "anchor_threshold": "AnchorThresholdValue",
    "scan_interval": "ScanInterval",
    "minimum_path_length": "MinPathLength",
    "sigma": "Sigma",
    "sum_gradient": "SumFlag",
    "validate_nfa": "NFAValidation",
    "minimum_line_length": "MinLineLength",
    "line_join_distance": "MaxDistanceBetweenTwoLines",
    "line_fit_error": "LineFitErrorThreshold",
    "maximum_line_error": "MaxErrorThreshold",
}


def main():
    settings = json.loads(sys.argv[1])
    cv2.setNumThreads(1)
    image = np.load("image.npy", allow_pickle=False)
    algorithm = cv2.ximgproc.createEdgeDrawing()
    native = cv2.ximgproc.EdgeDrawing.Params()
    native.EdgeDetectionOperator = getattr(cv2.ximgproc, "EdgeDrawing_" + settings["gradient"].upper())
    for field, name in _NATIVE_FIELDS.items():
        value = settings[field]
        setattr(native, name, type(getattr(native, name))(value))
    algorithm.setParams(native)
    algorithm.detectEdges(image)

    paths = algorithm.getSegments()
    lengths = np.array([len(path) for path in paths], np.int32)
    if len(paths) > MAX_PATHS or int(lengths.sum()) > MAX_PATH_POINTS:
        raise ValueError("Edge paths exceed the output budget. Raise gradient threshold or crop.")
    if settings["fit_lines"] and np.any(lengths > 8 * sum(image.shape[:2])):
        # The pinned OpenCV line fitter copies each path into fixed-size scratch arrays.
        raise ValueError("An edge path exceeds native line-fit capacity. Disable lines or crop.")
    edge, gradient = algorithm.getEdgeImage(), algorithm.getGradientImage()
    lines = algorithm.detectLines() if settings["fit_lines"] else None
    line_paths = algorithm.getSegmentIndicesOfLines() if lines is not None else ()
    shapes = algorithm.detectEllipses() if settings["fit_round"] else None
    if (lines is not None and len(lines) > 16_384) or (shapes is not None and len(shapes) > 4096):
        raise ValueError("Fitted primitives exceed the output budget. Reduce edge complexity or crop.")

    np.savez(
        "result.npz",
        edge=edge, gradient=gradient, lengths=lengths,
        points=np.concatenate(paths).reshape(-1, 2) if paths else np.empty((0, 2), np.int32),
        lines=np.asarray(lines, np.float32).reshape(-1, 4) if lines is not None else np.empty((0, 4), np.float32),
        line_paths=np.asarray(line_paths, np.int32).reshape(-1),
        shapes=np.asarray(shapes, np.float64).reshape(-1, 6) if shapes is not None else np.empty((0, 6), np.float64),
    )


if __name__ == "__main__":
    main()
