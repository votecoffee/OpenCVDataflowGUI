"""Measure detector repeatability against independent registration truth."""

import math

import numpy as np

from .core import Block, Port, Result
from .feature_benchmark import geometric_edges, in_support, maximum_matching, project, reference_matrix
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .selection import selection_mask
from .tickets import ticket_key

MAX_FEATURES = 4096


def observations(value, size):
    if not isinstance(value, dict):
        raise ValidationError("Supply a keypoint record or MSER regions")
    is_mser = value.get("model") == "mser_regions_v1"
    if not is_mser and value.get("image_size") != size:
        raise ValidationError("Keypoint record image size differs from its evaluation image")
    items = value.get("items" if is_mser else "keypoints")
    if not isinstance(items, list) or len(items) > MAX_FEATURES:
        raise ValidationError(f"Supply at most {MAX_FEATURES:,} observations per image")
    if value.get("truncated"):
        raise ValidationError("Repeatability needs complete observation records. Increase the detector limit.")
    try:
        points = np.asarray([x["center" if is_mser else "point"] for x in items], np.float64)
        sizes = np.asarray(
            [2 * math.sqrt(x["area"] / math.pi) if is_mser else x["size"] for x in items],
            np.float64,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Observations need finite centers and positive sizes") from exc
    if not items:
        points = points.reshape(0, 2)
    if (
        points.shape != (len(items), 2) or sizes.shape != (len(items),)
        or not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= size)
        or not np.isfinite(sizes).all() or np.any(sizes < 1e-6) or np.any(sizes > 1e6)
    ):
        raise ValidationError("Observations need in-bounds centers and sizes within 1e-6..1e6 pixels")
    method = "area_equivalent_diameter" if is_mser else "keypoint_size"
    return points, sizes, method


def similarity_scale(matrix):
    if np.max(np.abs(matrix[2, :2] / matrix[2, 2])) > 1e-12:
        return None
    linear = matrix[:2, :2] / matrix[2, 2]
    gram = linear.T @ linear
    scale_squared = float(np.trace(gram) / 2)
    if np.max(np.abs(gram - scale_squared * np.eye(2))) > 1e-8 * scale_squared:
        return None
    return math.sqrt(scale_squared)


class EvaluateKeypointRepeatability(Block):
    type_id, title = "inspect.evaluate_keypoints", "Evaluate keypoint repeatability"
    category = block_type = "Inspection"
    subtype = "Feature evaluation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("query_image"), Port("train_image"), Port("query", "data"), Port("train", "data"),
        Port("transform", "data", required=False, required_when_connected=True),
        Port("query_mask", required=False, required_when_connected=True),
        Port("train_mask", required=False, required_when_connected=True),
    )
    outputs = (Port("report", "data"), Port("repeatability", "float", required=False), Port("matches", "data"))
    defaults = {
        "reference_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "tolerance": 2.0,
        "minimum_pairs": 1,
        "minimum_repeatability": 0.0,
    }
    parameter_ranges = {
        "tolerance": NumericRange(1e-6, 1024),
        "minimum_pairs": NumericRange(1, MAX_FEATURES, whole=True),
        "minimum_repeatability": NumericRange(0, 1),
    }
    parameter_help = {
        "reference_transform": "Known query-to-train homography. A wired transform overrides it.",
        "tolerance": "Maximum center error in target pixels for one-to-one geometry matches.",
        "minimum_pairs": "Minimum one-to-one correspondence count to pass.",
        "minimum_repeatability": "Minimum pairs divided by the smaller eligible population.",
    }
    input_help = {
        "query": "Built-in keypoint list record or MSER region record for the query image.",
        "train": "Corresponding detector record from the train image.",
        "transform": "Independent query-to-train transform used as ground truth.",
    }
    output_help = {
        "report": "Population, center error and size ratio. Size error is evaluated on similarity transforms.",
        "repeatability": "Geometric correspondence count divided by the smaller eligible count.",
        "matches": "Ordered query/train centers and their frame contexts for Review feature matches.",
    }

    def process(self, inputs):
        observed, sizes, selections, contexts = {}, {}, {}, {}
        for role in ("query", "train"):
            name = role + "_image"
            image = inputs[name]
            sizes[role] = list(image.shape[1::-1])
            image_context = self.input_payloads[name]
            key_context = self.input_payloads[role]
            require_same_frame(
                [ValuePacket(None, image_context), ValuePacket(None, key_context)],
                message=f"{role.title()} observations use a different image frame",
            )
            if "ticket" in image_context and "ticket" in key_context:
                if ticket_key(image_context["ticket"]) != ticket_key(key_context["ticket"]):
                    raise ValidationError(f"{role.title()} observations belong to another ticket", role)
            observed[role] = observations(inputs[role], sizes[role])
            selections[role] = selection_mask({"mask": inputs.get(role + "_mask")}, image)
            contexts[role] = {
                field: image_context[field] for field in ("spatial", "ticket", "pyramid")
                if field in image_context
            }

        query, query_diameter, query_method = observed["query"]
        train, train_diameter, train_method = observed["train"]
        p = self.parameters
        matrix, inverse = reference_matrix(
            inputs.get("transform", p["reference_transform"]), sizes["query"], sizes["train"],
        )
        projected, back = project(query, matrix), project(train, inverse)
        qkeep = in_support(query, sizes["query"], selections["query"], 0) & in_support(
            projected, sizes["train"], selections["train"], 0,
        )
        tkeep = in_support(train, sizes["train"], selections["train"], 0) & in_support(
            back, sizes["query"], selections["query"], 0,
        )
        if int(qkeep.sum()) * int(tkeep.sum()) > 4_000_000:
            raise ValidationError("Repeatability exceeds four million eligible pairs")
        adjacency = geometric_edges(projected, train, qkeep, tkeep, float(p["tolerance"]), 100_000)
        pairing = maximum_matching(adjacency, len(train))
        qidx = np.array(list(pairing), dtype=np.intp)
        tidx = np.array(list(pairing.values()), dtype=np.intp)
        smaller = int(min(qkeep.sum(), tkeep.sum()))
        repeatability = len(pairing) / smaller if smaller else None
        passed = len(pairing) >= p["minimum_pairs"] and repeatability >= p["minimum_repeatability"]
        errors = np.linalg.norm(projected[qidx] - train[tidx], axis=1)
        ratios = train_diameter[tidx] / query_diameter[qidx]
        scale = similarity_scale(matrix)
        relative = np.abs(ratios / scale - 1) if scale is not None else np.empty(0)
        info = {
            "query_count": len(query), "train_count": len(train),
            "query_eligible": int(qkeep.sum()), "train_eligible": int(tkeep.sum()),
            "correspondence_count": len(pairing), "repeatability": repeatability,
            "median_center_error_px": float(np.median(errors)) if errors.size else None,
            "p95_center_error_px": float(np.quantile(errors, 0.95)) if errors.size else None,
            "query_size_method": query_method, "train_size_method": train_method,
            "expected_similarity_scale": scale,
            "median_size_ratio": float(np.median(ratios)) if ratios.size else None,
            "p95_relative_size_error": float(np.quantile(relative, 0.95)) if relative.size else None,
        }
        return Result(
            {
                "report": info,
                "repeatability": repeatability if repeatability is not None else NO_VALUE,
                "matches": {
                    "query_size": sizes["query"], "train_size": sizes["train"],
                    "query_points": query[qidx].tolist(), "train_points": train[tidx].tolist(),
                    "query_context": contexts["query"], "train_context": contexts["train"],
                },
            },
            passed=passed,
            score=repeatability,
            fail_reasons=[] if passed else ["too_few_repeatable_keypoints"],
            message=f"{len(pairing)}/{smaller} repeatable centers",
        )
