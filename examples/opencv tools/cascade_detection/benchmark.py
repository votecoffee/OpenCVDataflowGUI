"""Independent direct-native reference; imports no application processing code."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def measure():
    truth = json.loads((HERE / "truth.json").read_bytes())
    cases = {}
    for case, expected in truth["cases"].items():
        if case == "bad_pin":
            continue
        project = json.loads((HERE / (case + ".json")).read_bytes())
        p = next(b["parameters"] for b in project["blocks"] if b["id"] == "detect")
        image = cv2.imread(str(HERE / "assets" / expected["image"]), cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.dtype == np.uint16:
            gray = np.rint(gray.astype(float) / 65535 * 255).astype(np.uint8)
        if p["equalize_histogram"]:
            gray = cv2.equalizeHist(gray)
        model = cv2.CascadeClassifier(str(HERE / p["path"]))
        kwargs = dict(
            scaleFactor=p["scale_factor"],
            minNeighbors=p["min_neighbors"],
            flags=0,
            minSize=tuple(p["min_size"]),
            maxSize=tuple(p["max_size"]),
        )
        mode = p["result_mode"]
        if mode == "weights":
            boxes, rejects, scores = model.detectMultiScale3(
                gray, outputRejectLevels=True, **kwargs
            )
        elif mode == "neighbors":
            boxes, scores = model.detectMultiScale2(gray, **kwargs)
            rejects = [None] * len(boxes)
        else:
            boxes = model.detectMultiScale(gray, **kwargs)
            rejects, scores = [None] * len(boxes), [None] * len(boxes)
        rows = []
        for box, score, reject in zip(boxes, scores, rejects, strict=True):
            x, y, w, h = map(int, box)
            if p["strict_size"] and any(
                n < lo or (hi and n > hi)
                for n, lo, hi in zip([w, h], p["min_size"], p["max_size"], strict=True)
            ):
                continue
            if p["minimum_weight_enabled"] and score < p["minimum_weight"]:
                continue
            if case == "selection" and x + (w - 1) / 2 >= 512:
                continue
            rows.append(
                {
                    "box": [x, y, w, h],
                    "score": None if score is None else float(score),
                    "reject_level": None if reject is None else int(reject),
                }
            )
        rows.sort(key=lambda row: (*row["box"], row["score"] or 0))
        cases[case] = {"native_candidates": len(boxes), "retained": rows}
    return {
        "opencv": cv2.__version__,
        "scope": "Direct native parity plus independent score/size/selection filtering, "
        "not independent cascade inference or field accuracy",
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), sort_keys=True, indent=2, allow_nan=False))
