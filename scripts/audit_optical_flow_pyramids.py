"""Probe Python pyramid construction/consumption without claiming a cached tracker."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def audit():
    image = np.random.default_rng(155).integers(0, 256, (129, 161), np.uint8)
    before = image.copy()
    points = np.array([[[40.0, 40.0]], [[80.0, 60.0]]], np.float32)
    result = {
        "model": "optical_flow_pyramid_binding_audit_v1",
        "opencv": cv2.__version__,
        "construction": [],
        "consumption": [],
    }
    for derivatives in (False, True):
        level, pyramid = cv2.buildOpticalFlowPyramid(
            image,
            (21, 21),
            3,
            withDerivatives=derivatives,
            pyrBorder=cv2.BORDER_REFLECT_101,
            derivBorder=cv2.BORDER_CONSTANT,
            tryReuseInputImage=True,
        )
        result["construction"].append(
            {
                "with_derivatives": derivatives,
                "requested_max_level": 3,
                "actual_max_level": level,
                "window": [21, 21],
                "pyramid_border": "reflect_101",
                "derivative_border": "constant",
                "requested_input_reuse": True,
                "python_level0_shares_input": bool(np.shares_memory(image, pyramid[0])),
                "arrays": [
                    {"shape": list(array.shape), "dtype": str(array.dtype)} for array in pyramid
                ],
            }
        )
        for name, value in (
            ("tuple", pyramid),
            ("list", list(pyramid)),
            ("umat_tuple", tuple(cv2.UMat(array) for array in pyramid)),
        ):
            for api in ("function", "object"):
                record = {"with_derivatives": derivatives, "representation": name, "api": api}
                function = (
                    cv2.calcOpticalFlowPyrLK
                    if api == "function"
                    else cv2.SparsePyrLKOpticalFlow.create().calc
                )
                try:
                    found, status, _ = function(value, value, points, None)
                    record.update(
                        accepted=True,
                        identity_points_match=bool(
                            found is not None
                            and np.all(status)
                            and np.allclose(found, points, atol=0.01)
                        ),
                    )
                except cv2.error as exc:
                    record.update(accepted=False, error=str(exc)[-512:])
                result["consumption"].append(record)
    result["input_unchanged"] = bool(np.array_equal(image, before))
    result["python_cached_consumer_qualified"] = any(
        entry.get("identity_points_match", False) for entry in result["consumption"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Create a new JSON receipt")
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
