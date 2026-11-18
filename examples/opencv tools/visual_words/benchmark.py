"""Independent native extraction, NumPy quantization and gallery ranking references."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def descriptors(image, method, settings):
    if method == "sift":
        detector = cv2.SIFT_create(
            nfeatures=settings["max_features"],
            nOctaveLayers=settings["sift_octave_layers"],
            contrastThreshold=settings["sift_contrast_threshold"],
            edgeThreshold=settings["sift_edge_threshold"],
            sigma=settings["sift_sigma"],
            enable_precise_upscale=settings["sift_precise_upscale"],
        )
    else:
        detector = cv2.ORB_create(
            nfeatures=settings["max_features"],
            scaleFactor=settings["orb_scale_factor"],
            nlevels=settings["orb_levels"],
            edgeThreshold=settings["orb_edge_threshold"],
            firstLevel=settings["orb_first_level"],
            WTA_K=int(settings["orb_wta_k"]),
            scoreType=cv2.ORB_HARRIS_SCORE
            if settings["orb_score"] == "harris"
            else cv2.ORB_FAST_SCORE,
            patchSize=settings["orb_patch_size"],
            fastThreshold=settings["orb_fast_threshold"],
        )
    keys, rows = detector.detectAndCompute(image, None)
    order = sorted(
        range(len(keys)),
        key=lambda i: (-keys[i].response, *keys[i].pt, keys[i].size, keys[i].angle, keys[i].octave),
    )[: settings["max_features"]]
    return rows[order]


def measure():
    truth = json.loads((HERE / "truth.json").read_bytes())
    cases = {}
    for method, normalization in (("sift", "l1"), ("orb", "l1"), ("sift", "hellinger")):
        codebook = json.loads((HERE / f"assets/codebook-{method}.json").read_bytes())
        gallery = json.loads((HERE / f"assets/gallery-{method}-{normalization}.json").read_bytes())
        dtype, norm = (
            (np.float32, cv2.NORM_L2) if method == "sift" else (np.uint8, cv2.NORM_HAMMING)
        )
        centers = np.asarray(codebook["centers"], dtype)
        queries = []
        for name in [*truth["queries"], "rearranged.png"]:
            image = cv2.imread(str(HERE / "assets" / name), cv2.IMREAD_GRAYSCALE)
            rows = descriptors(image, method, codebook["extractor"]["settings"])
            if method == "sift":
                distances = np.linalg.norm(
                    rows[:, None].astype(float) - centers[None].astype(float), axis=2
                )
            else:
                distances = np.unpackbits(np.bitwise_xor(rows[:, None], centers[None]), axis=2).sum(
                    2
                )
            ids = distances.argmin(1)
            matches = cv2.BFMatcher(norm).match(rows, centers)
            native_ids = np.array(
                [match.trainIdx for match in sorted(matches, key=lambda m: m.queryIdx)]
            )
            histogram = np.bincount(ids, minlength=len(centers)).astype(float)
            histogram /= histogram.sum()
            if normalization == "hellinger":
                histogram = np.sqrt(histogram)
            ranked = []
            for entry in gallery["entries"]:
                ref = np.asarray(entry["histogram"])
                distance = float(
                    np.clip(
                        1 - (histogram @ ref) / np.linalg.norm(histogram) / np.linalg.norm(ref),
                        0,
                        1,
                    )
                )
                ranked.append({"label": entry["label"], "distance": distance})
            ranked.sort(key=lambda row: row["distance"])
            queries.append(
                {
                    "image": name,
                    "descriptors": len(rows),
                    "assignment_disagreements": int(np.count_nonzero(ids != native_ids)),
                    "maximum_distance_error": float(
                        np.max(
                            np.abs(
                                distances[np.arange(len(rows)), ids]
                                - [m.distance for m in sorted(matches, key=lambda m: m.queryIdx)]
                            )
                        )
                    ),
                    "histogram": histogram.tolist(),
                    "ranking": ranked,
                }
            )
        cases[f"{method}-{normalization}"] = queries
    return {
        "opencv": cv2.__version__,
        "reference": "native SIFT/ORB and independent NumPy Euclidean/bit-count quantization "
        "and cosine ranking",
        "scope": "Original controlled held-out views; no pose, layout or production accuracy claim",
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, allow_nan=False))
