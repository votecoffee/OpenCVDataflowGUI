"""Ordered single-object landmarks, explicit heatmap geometry and local inspection poses."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .detection_decode import check_context, preprocessing_geometry, probability
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import selected_tensor
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .poses import POSE_HELP, POSE_OFFSETS, local_offset, make_pose, offset_pose, pose_for_image
from .processing import image_value
from .selection import contains, selection_mask
from .tickets import ticket_key

MODEL = "image_landmarks_v1"


def tensor(inputs, p):
    try:
        records = inputs["tensors"]
        chosen = records[p["output_name"]] if p["output_name"] else next(iter(records.values()))
        if not isinstance(chosen, dict) or chosen.get("dtype") != "float32":
            raise ValidationError("Use dequantized float32 tensor_v1 landmark outputs")
        flat = chosen.get("values")
        if (
            not isinstance(flat, list)
            or len(flat) > p["maximum_tensor_elements"]
            or any(type(v) not in (int, float) for v in flat)
        ):
            raise ValidationError("Landmark values must be a bounded flat numeric list")
        with np.errstate(over="ignore", invalid="ignore"):
            return selected_tensor(
                records,
                p["output_name"],
                max_elements=p["maximum_tensor_elements"],
                allow_empty=True,
            ).astype(np.float64)
    except ValidationError:
        raise
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Invalid landmark tensor: " + str(exc)) from exc


def network_edges(points, head_size, network_size, mapping):
    points, head, network = (
        np.asarray(points, np.float64),
        np.array(head_size),
        np.array(network_size),
    )
    if mapping == "half_pixel":
        return (points + 0.5) * network / head
    if mapping == "asymmetric":
        return points * network / head + 0.5
    scale = (network - 1) / np.maximum(1, head - 1)
    return np.where(head == 1, network / 2, points * scale + 0.5)


def refine(plane, x, y, eligible, p):
    if p["refinement"] == "none":
        return np.array([x, y], float), "disabled", None
    r = p["refinement_radius"]
    if min(x, y) < r or x + r >= plane.shape[1] or y + r >= plane.shape[0]:
        return np.array([x, y], float), "boundary", None
    if eligible is not None and not eligible[y - r : y + r + 1, x - r : x + r + 1].all():
        return np.array([x, y], float), "selection_boundary", None
    window = plane[y - r : y + r + 1, x - r : x + r + 1]
    if p["refinement"] == "log_quadratic":
        if np.any(window <= 0):
            return np.array([x, y], float), "nonpositive_window", None
        window = np.log(window)
    v, u = np.mgrid[-r : r + 1, -r : r + 1]
    design = np.column_stack(
        (u.ravel() ** 2, (u * v).ravel(), v.ravel() ** 2, u.ravel(), v.ravel(), np.ones(u.size))
    )
    coefficients = np.linalg.lstsq(design, window.ravel(), rcond=None)[0]
    a, b, c, d, e, _ = coefficients
    hessian = np.array([[2 * a, b], [b, 2 * c]])
    rmse = float(np.sqrt(np.mean((design @ coefficients - window.ravel()) ** 2)))
    if np.max(np.linalg.eigvalsh(hessian)) >= -p["minimum_curvature"]:
        return np.array([x, y], float), "not_concave", rmse
    if p["maximum_fit_rmse"] and rmse > p["maximum_fit_rmse"]:
        return np.array([x, y], float), "poor_fit", rmse
    shift = np.linalg.solve(hessian, -np.array([d, e]))
    if not np.isfinite(shift).all() or np.max(np.abs(shift)) > p["maximum_refinement_offset"]:
        return np.array([x, y], float), "offset_limit", rmse
    return np.array([x, y], float) + shift, "refined", rmse


class DecodeLandmarks(Block):
    type_id, title = "ml.decode_landmarks", "Decode keypoints and heatmaps"
    category, block_type, subtype = "ML", "ML", "Landmarks"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("image"),
        Port("tensors", "data"),
        Port("preprocessing", "data"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("landmarks", "data"),
        Port("points", "data", required=False),
        Port("indices", "data"),
        Port("count", "integer"),
        Port("report", "data"),
        Port("object_confidence", "number", required=False),
    )
    defaults = {
        "output_name": "",
        "format": "heatmaps",
        "layout": "NCHW",
        "channels": [],
        "names": [],
        "score_activation": "probability",
        "minimum_confidence": 0.5,
        "heatmap_mapping": "half_pixel",
        "peak_search": "selection",
        "ambiguity": "reject",
        "refinement": "log_quadratic",
        "refinement_radius": 1,
        "refinement_failure": "integer",
        "minimum_curvature": 1e-6,
        "maximum_refinement_offset": 0.75,
        "maximum_fit_rmse": 0.1,
        "coordinate_layout": "rows",
        "coordinate_columns": 3,
        "coordinate_xy_columns": [0, 1],
        "coordinate_score_columns": [2],
        "coordinate_score_combination": "minimum",
        "coordinate_units": "network_pixels",
        "coordinate_origin": "pixel_centers",
        "object_score_name": "",
        "object_score_activation": "probability",
        "minimum_object_confidence": 0.5,
        "minimum_count": 1,
        "require_all": True,
        "points_policy": "complete_only",
        "connections": [],
        "rejected_graphics": "show",
        "maximum_tensor_elements": 8_000_000,
        "maximum_landmarks": 1024,
        "maximum_heatmap_pixels": 1_048_576,
        "maximum_image_pixels": 4_194_304,
        "graphics_maximum_landmarks": 256,
    }
    parameter_choices = {
        "format": ("heatmaps", "xy_score"),
        "layout": ("NCHW", "NHWC", "CHW", "HWC"),
        "score_activation": ("probability", "sigmoid"),
        "heatmap_mapping": ("half_pixel", "align_corners", "asymmetric"),
        "peak_search": ("selection", "entire_head"),
        "ambiguity": ("reject", "first"),
        "refinement": ("none", "quadratic", "log_quadratic"),
        "refinement_failure": ("integer", "reject"),
        "coordinate_layout": ("rows", "channels_first", "flat_rows"),
        "coordinate_score_combination": ("minimum", "product", "mean"),
        "coordinate_units": ("network_pixels", "normalized"),
        "coordinate_origin": ("pixel_centers", "pixel_edges", "scaled_indices"),
        "object_score_activation": ("probability", "sigmoid"),
        "rejected_graphics": ("show", "hide"),
        "points_policy": ("complete_only", "accepted_only"),
    }
    parameter_ranges = {
        "minimum_confidence": NumericRange(0, 1),
        "minimum_object_confidence": NumericRange(0, 1),
        "coordinate_columns": NumericRange(3, 16, whole=True),
        "refinement_radius": NumericRange(1, 3, whole=True),
        "minimum_curvature": NumericRange(1e-12, 100),
        "maximum_refinement_offset": NumericRange(0.01, 3),
        "maximum_fit_rmse": NumericRange(0, 100),
        "minimum_count": NumericRange(0, 1024, whole=True),
        "maximum_tensor_elements": NumericRange(1, 8_000_000, whole=True),
        "maximum_landmarks": NumericRange(1, 1024, whole=True),
        "maximum_heatmap_pixels": NumericRange(1, 4_194_304, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "graphics_maximum_landmarks": NumericRange(1, 1024, whole=True),
    }
    parameter_help = {
        "output_name": "Named float32 model head; empty selects the first output.",
        "format": "One ordered peak per heatmap channel, or explicit ordered X/Y/score rows.",
        "layout": "Heatmap axes, with image batch N=1 when present.",
        "channels": (
            "Ordered distinct channel/row IDs; empty uses all. This can exclude background maps."
        ),
        "names": "Optional names in configured channel order; one name per selected landmark.",
        "score_activation": (
            "Declare probabilities, or convert logits with sigmoid before peak/confidence tests."
        ),
        "minimum_confidence": (
            "Accept confidence >= this threshold; heatmaps use the original integer-peak score."
        ),
        "heatmap_mapping": (
            "Declared head-to-network alignment. One-cell align-corners axes use the field center."
        ),
        "peak_search": (
            "Search eligible head-cell centers in source/ROI, or find the whole-head "
            "peak then filter it."
        ),
        "ambiguity": (
            "Reject equal global maxima, or explicitly select the first in row-major order."
        ),
        "refinement": (
            "Fit a concave quadratic to local scores or log scores; none retains "
            "integer peak cells."
        ),
        "refinement_radius": (
            "Square fit window radius in heatmap cells; source ROI boundaries are respected."
        ),
        "refinement_failure": (
            "Keep integer position or reject when a requested fit cannot be qualified."
        ),
        "minimum_curvature": "Minimum negative Hessian curvature magnitude for an accepted fit.",
        "maximum_refinement_offset": "Largest absolute fitted shift on either heatmap axis.",
        "maximum_fit_rmse": (
            "Maximum fit residual in score/log-score units; zero disables this gate."
        ),
        "coordinate_layout": (
            "Rows [K,C], channels-first [C,K], or one flat K*C row; optional batch one."
        ),
        "coordinate_columns": "Columns per direct landmark; unused auxiliary columns remain "
        "only in the input tensor.",
        "coordinate_xy_columns": "Two distinct direct-row column IDs for X/Y.",
        "coordinate_score_columns": "Distinct confidence columns, separate from X/Y; "
        "each is activated before combining.",
        "coordinate_score_combination": "Combine activated scores by minimum, product or mean; "
        "not calibrated correctness.",
        "coordinate_units": (
            "Direct coordinates in network pixels, or fractions multiplied by network width/height."
        ),
        "coordinate_origin": (
            "Pixel centers/edges use resize geometry. Scaled indices uses x*scale+offset "
            "without a half-pixel shift, for exports declaring that convention."
        ),
        "object_score_name": "Optional scalar float32 object-confidence head in this activation; "
        "empty disables.",
        "object_score_activation": "Separate object-head activation, independent of point scores.",
        "minimum_object_confidence": "Inclusive object gate; low scores reject all points.",
        "rejected_graphics": "Show diagnostic rejected points, or hide them. "
        "Links need two accepted endpoints.",
        "minimum_count": "Fewest accepted landmarks required for a normal pass judgment.",
        "require_all": (
            "Also fail if any requested landmark is rejected; identities are never renumbered."
        ),
        "points_policy": (
            "Default suppresses Points unless all requested landmarks are accepted; "
            "otherwise use Indices."
        ),
        "connections": (
            "Optional unique [index,index] pairs in configured landmark order for drawable links."
        ),
        "maximum_tensor_elements": "Bound serialized values before conversion.",
        "maximum_landmarks": "Bound selected landmarks; raw heads allow up to 1024 channels/rows.",
        "maximum_heatmap_pixels": "Bound each head plane, including zero-channel heads.",
        "maximum_image_pixels": "Bound source image pixels; axes are limited to 16384.",
        "graphics_maximum_landmarks": (
            "Limit optional markers/labels/links without truncating numerical results."
        ),
    }
    input_help = {
        "image": "Original source image; pixels/encoding/depth remain unchanged.",
        "tensors": "One object's ordered landmark head from this image's inference activation.",
        "preprocessing": "Matching inference resize/letterbox dimensions and inverse maps.",
        "mask": (
            "Optional source-grid eligibility/filter; a wired missing value prevents execution."
        ),
        "region": "Optional circle/rectangle/polygon/raster selection intersected with Mask.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [
            name for name in self.exposed_outputs if name != "object_confidence"
        ]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError("Use typed numeric landmark settings", name)
        if p["maximum_refinement_offset"] > p["refinement_radius"]:
            raise ValidationError(
                "Refinement offset exceeds the fit window",
                "maximum_refinement_offset",
                "refinement_radius",
            )
        for field in ("output_name", "object_score_name"):
            if not isinstance(p[field], str) or len(p[field]) > 256:
                raise ValidationError("Output name must be bounded text", field)
        xy, scores, columns = (
            p["coordinate_xy_columns"],
            p["coordinate_score_columns"],
            p["coordinate_columns"],
        )
        if (
            not isinstance(xy, list)
            or len(xy) != 2
            or any(type(n) is not int or not 0 <= n < columns for n in xy)
            or xy[0] == xy[1]
        ):
            raise ValidationError(
                "X/Y columns must be two distinct valid integer IDs", "coordinate_xy_columns"
            )
        if (
            not isinstance(scores, list)
            or not 1 <= len(scores) <= 14
            or any(type(n) is not int or not 0 <= n < columns for n in scores)
            or len(set(scores)) != len(scores)
            or set(scores) & set(xy)
        ):
            raise ValidationError(
                "Score columns must be unique valid IDs separate from X/Y",
                "coordinate_score_columns",
            )
        if type(p["require_all"]) is not bool:
            raise ValidationError("Require all is a checkbox", "require_all")
        if (
            not isinstance(p["channels"], list)
            or len(p["channels"]) > 1024
            or any(type(n) is not int or not 0 <= n < 1024 for n in p["channels"])
            or len(set(p["channels"])) != len(p["channels"])
        ):
            raise ValidationError("Channels must be unique integer IDs in 0..1023", "channels")
        if (
            not isinstance(p["names"], list)
            or len(p["names"]) > 1024
            or any(not isinstance(s, str) or not s or len(s) > 256 for s in p["names"])
        ):
            raise ValidationError("Names must be a bounded list of nonempty strings", "names")
        pairs = p["connections"]
        if (
            not isinstance(pairs, list)
            or len(pairs) > 2048
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(n) is not int or not 0 <= n < 1024 for n in pair)
                or pair[0] == pair[1]
                for pair in pairs
            )
        ):
            raise ValidationError(
                "Connections need bounded distinct-endpoint integer pairs", "connections"
            )
        if len({tuple(sorted(pair)) for pair in pairs}) != len(pairs):
            raise ValidationError("Connections must be unique", "connections")

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        h, w = source.shape[:2]
        if min(h, w) < 1 or max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError(
                "Source exceeds landmark pixel/axis budget", "maximum_image_pixels"
            )
        image_value(source)
        checked = check_context(self, kind="Landmark")
        network, reverse = preprocessing_geometry(inputs["preprocessing"], source)
        selection = selection_mask(inputs, source)
        values = tensor(inputs, p)
        object_confidence, object_ok = NO_VALUE, True
        if p["object_score_name"]:
            object_values = tensor(inputs, {**p, "output_name": p["object_score_name"]})
            if object_values.size != 1:
                raise ValidationError("Object confidence requires one scalar", "object_score_name")
            if values.size + 1 > p["maximum_tensor_elements"]:
                raise ValidationError(
                    "Combined landmark/object tensors exceed the element budget",
                    "maximum_tensor_elements",
                )
            object_confidence = float(
                probability(object_values, p["object_score_activation"]).flat[0]
            )
            object_ok = object_confidence >= p["minimum_object_confidence"]
        heatmaps = p["format"] == "heatmaps"
        if heatmaps:
            dims = 4 if p["layout"].startswith("N") else 3
            if values.ndim != dims or dims == 4 and values.shape[0] != 1:
                raise ValidationError(
                    "Heatmap layout requires one image and explicit axes", "layout"
                )
            if dims == 4:
                values = values[0]
            if p["layout"].endswith("HWC"):
                values = values.transpose(2, 0, 1)
            k, hh, hw = values.shape
            if not 1 <= min(hh, hw) or max(hh, hw) > 4096 or hh * hw > p["maximum_heatmap_pixels"]:
                raise ValidationError(
                    "Heatmap plane exceeds pixel/axis budget", "maximum_heatmap_pixels"
                )
        else:
            if values.ndim == 3 and values.shape[0] == 1:
                values = values[0]
            if p["coordinate_layout"] == "flat_rows":
                if values.ndim != 1 and not (values.ndim == 2 and values.shape[0] == 1):
                    raise ValidationError("Flat landmarks need a vector or a single flat row")
                if values.size % p["coordinate_columns"]:
                    raise ValidationError(
                        "Flat landmark size is not a multiple of Coordinate columns"
                    )
                values = values.reshape(-1, p["coordinate_columns"])
            else:
                if values.ndim != 2:
                    raise ValidationError("Direct landmarks need one image's coordinate/score rows")
                if p["coordinate_layout"] == "channels_first":
                    values = values.T
            if values.shape[1] != p["coordinate_columns"]:
                raise ValidationError("Direct row width differs from Coordinate columns")
            k = len(values)
        channels = list(p["channels"]) if p["channels"] else list(range(k))
        if k > 1024 or len(channels) > p["maximum_landmarks"] or any(n >= k for n in channels):
            raise ValidationError(
                "Channels/landmark count outside the head/budget", "channels", "maximum_landmarks"
            )
        if p["names"] and len(p["names"]) != len(channels):
            raise ValidationError("Supply one name per selected landmark", "names")
        if any(max(pair) >= len(channels) for pair in p["connections"]):
            raise ValidationError(
                "Connection index outside configured landmark order", "connections"
            )
        eligible = None
        if heatmaps and p["peak_search"] == "selection":
            grid = network_edges(
                np.stack(np.meshgrid(np.arange(hw), np.arange(hh)), axis=-1),
                [hw, hh],
                network,
                p["heatmap_mapping"],
            )
            grid = grid * [reverse[0, 0], reverse[1, 1]] + [
                reverse[0, 2] - 0.5,
                reverse[1, 2] - 0.5,
            ]
            eligible = (
                (grid[..., 0] >= 0)
                & (grid[..., 0] <= w - 1)
                & (grid[..., 1] >= 0)
                & (grid[..., 1] <= h - 1)
            )
            if selection is not None:
                ix, iy = (
                    np.rint(grid[..., 0]).astype(int).clip(0, w - 1),
                    np.rint(grid[..., 1]).astype(int).clip(0, h - 1),
                )
                eligible &= selection[iy, ix] != 0
        components = None
        if heatmaps:
            scores = probability(values, p["score_activation"])
        else:
            components = probability(
                values[:, p["coordinate_score_columns"]], p["score_activation"]
            )
            combine = {"minimum": np.min, "product": np.prod, "mean": np.mean}
            scores = combine[p["coordinate_score_combination"]](components, axis=1)
        items = []
        for index, channel in enumerate(channels):
            status, rmse, ambiguous, position, model_position = "direct", None, False, None, None
            confidence, reason = None, ""
            if heatmaps:
                plane = scores[channel]
                if eligible is not None and not eligible.any():
                    reason, status = "no_eligible_peak", "unavailable"
                else:
                    _, confidence, _, (x, y) = cv2.minMaxLoc(
                        plane, mask=None if eligible is None else eligible.astype(np.uint8)
                    )
                    ambiguous = (
                        np.count_nonzero(
                            (plane == confidence) & (True if eligible is None else eligible)
                        )
                        > 1
                    )
                    refined, status, rmse = refine(plane, x, y, eligible, p)
                    model_position = network_edges(refined, [hw, hh], network, p["heatmap_mapping"])
                    if ambiguous and p["ambiguity"] == "reject":
                        reason = "ambiguous_peak"
                    elif p["refinement_failure"] == "reject" and status not in (
                        "disabled",
                        "refined",
                    ):
                        reason = "refinement_rejected"
            else:
                model_position = values[channel, p["coordinate_xy_columns"]].copy()
                confidence = float(scores[channel])
                if p["coordinate_units"] == "normalized":
                    model_position *= network
                if p["coordinate_origin"] == "pixel_centers":
                    model_position += 0.5
            if model_position is not None:
                origin_shift = (
                    0 if not heatmaps and p["coordinate_origin"] == "scaled_indices" else -0.5
                )
                position = model_position * [reverse[0, 0], reverse[1, 1]] + [
                    reverse[0, 2] + origin_shift,
                    reverse[1, 2] + origin_shift,
                ]
                if np.any(np.abs(position) > 1e7) or not np.isfinite(position).all():
                    raise ValidationError("Landmark coordinates exceed finite source limits")
                if not reason and not (0 <= position[0] <= w - 1 and 0 <= position[1] <= h - 1):
                    reason = "outside_source"
                if not reason and not contains(selection, position):
                    reason = "outside_selection"
                if not reason and confidence < p["minimum_confidence"]:
                    reason = "low_confidence"
            if not object_ok:
                reason = "low_object_confidence"
            items.append(
                {
                    "index": index,
                    "channel": channel,
                    "name": p["names"][index] if p["names"] else str(channel),
                    "accepted": not bool(reason),
                    "position": position.tolist() if position is not None else None,
                    "confidence": confidence,
                    "reason": reason,
                    "refinement": status,
                    "fit_rmse": rmse,
                    "ambiguous": bool(ambiguous),
                }
            )
            if components is not None:
                items[-1]["confidence_components"] = components[channel].tolist()
        accepted = [i for i in items if i["accepted"]]
        count = len(accepted)
        passed = (
            object_ok
            and count >= p["minimum_count"]
            and (not p["require_all"] or count == len(items))
        )
        context = self.input_payloads.get("image", {})
        output_context = {
            key: copy.deepcopy(context[key])
            for key in ("spatial", "pyramid", "ticket", "recipe_ref")
            if key in context
        }
        data = {
            "model": MODEL,
            "image_size": [w, h],
            "coordinate_convention": "source_pixel_centers",
            "items": items,
            "context": output_context,
        }
        complete = count == len(items)
        points = (
            [i["position"] for i in accepted]
            if complete or p["points_policy"] == "accepted_only"
            else NO_VALUE
        )
        objects = []
        drawn_points, drawn_links = [], []
        limit = p["graphics_maximum_landmarks"]
        if self.generate_graphics:
            for item in items[:limit]:
                if not item["accepted"] and p["rejected_graphics"] == "hide":
                    continue
                xy = item["position"]
                if xy is None or not 0 <= xy[0] <= w - 1 or not 0 <= xy[1] <= h - 1:
                    continue
                color = (70, 220, 130) if item["accepted"] else (60, 80, 240)
                drawn_points.append(item["index"])
                objects.extend(
                    (
                        graphic_object("marker", [xy], color),
                        graphic_object(
                            "label",
                            [[xy[0] + 4, xy[1]]],
                            color,
                            text=item["name"]
                            + (
                                f" {item['confidence']:.3f}"
                                if item["confidence"] is not None
                                else ""
                            ),
                        ),
                    )
                )
            for a, b in p["connections"]:
                if max(a, b) < limit and items[a]["accepted"] and items[b]["accepted"]:
                    drawn_links.append([a, b])
                    objects.append(
                        graphic_object(
                            "polyline", [items[a]["position"], items[b]["position"]], (80, 180, 240)
                        )
                    )
        report = {
            "model": "landmark_decode_v1",
            "format": p["format"],
            "channels": channels,
            "requested": len(items),
            "accepted": count,
            "complete": complete,
            "points_policy": p["points_policy"],
            "context_checks": checked,
            "heatmap_mapping": p["heatmap_mapping"] if heatmaps else None,
            "graphics_truncated": self.generate_graphics and len(items) > limit,
            "drawn_points": drawn_points,
            "drawn_links": drawn_links,
            "coordinate_columns": p["coordinate_columns"] if not heatmaps else None,
            "score_columns": p["coordinate_score_columns"] if not heatmaps else None,
            "score_combination": p["coordinate_score_combination"] if not heatmaps else None,
            "object_confidence": None if object_confidence is NO_VALUE else object_confidence,
            "object_accepted": object_ok,
        }
        outputs = {
            "image": source,
            "landmarks": data,
            "points": points,
            "indices": [i["index"] for i in accepted],
            "count": count,
            "report": report,
            "object_confidence": object_confidence,
        }
        return Result(
            outputs,
            payloads={key: copy.deepcopy(context) for key in outputs},
            passed=passed,
            score=min((i["confidence"] for i in accepted), default=0),
            message=f"{count}/{len(items)} landmarks accepted",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "landmark.object_confidence" if not object_ok else "landmark.count",
                    "message": (
                        f"Object confidence {object_confidence:.6g} below "
                        f"{p['minimum_object_confidence']:.6g}"
                        if not object_ok
                        else "Required landmarks or minimum accepted count not met"
                    ),
                }
            ],
            drawings={"Landmarks": GraphicDrawing(tuple(objects))}
            if self.generate_graphics
            else {},
        )


class LandmarkPose(Block):
    type_id, title = "ml.landmark_pose", "Pose from ordered landmarks"
    category, block_type, subtype = "ML", "ML", "Landmarks"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("image"), Port("landmarks", "data"))
    outputs = (
        Port("image"),
        Port("pose", "data", required=False, visual_type="pose"),
        Port("found", "boolean"),
    )
    defaults = {
        "origin_index": 0,
        "angle_mode": "pair",
        "direction_index": 1,
        "angle_degrees": 0.0,
        "minimum_baseline": 1.0,
        **POSE_OFFSETS,
    }
    parameter_choices = {"angle_mode": ("pair", "fixed")}
    parameter_ranges = {
        **{k: NumericRange(-1e6, 1e6) for k in POSE_OFFSETS},
        **{k: NumericRange(0, 1023, whole=True) for k in ("origin_index", "direction_index")},
        "minimum_baseline": NumericRange(1e-6, 1e6),
        "angle_degrees": NumericRange(-1e6, 1e6),
    }
    parameter_help = {
        **POSE_HELP,
        "origin_index": "Stable configured landmark index, including rejected/missing entries.",
        "angle_mode": "Pair uses origin-to-direction axis; fixed uses counterclockwise angle.",
        "direction_index": "Second accepted landmark defining the local positive x direction.",
        "angle_degrees": "Fixed counterclockwise degrees, in the source landmark image grid.",
        "minimum_baseline": (
            "Minimum source-pixel separation before a landmark pair defines an axis."
        ),
    }
    input_help = {
        "image": "Target image; shared-root transforms map the source landmark frame when needed.",
        "landmarks": "Matching image_landmarks_v1 data with stable indices and source context.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError("Use typed numeric pose settings", name)
        if p["angle_mode"] == "pair" and p["origin_index"] == p["direction_index"]:
            raise ValidationError(
                "Origin and direction indices must differ", "origin_index", "direction_index"
            )

    def process(self, inputs):
        source, data, p = inputs["image"], inputs["landmarks"], self.parameters
        image_value(source)
        if (
            not isinstance(data, dict)
            or data.get("model") != MODEL
            or data.get("coordinate_convention") != "source_pixel_centers"
            or not isinstance(data.get("items"), list)
            or len(data["items"]) > 1024
        ):
            raise ValidationError("Connect ordered image_landmarks_v1 results")
        body, wire = data.get("context", {}), self.input_payloads.get("landmarks", {})
        require_same_frame(
            [ValuePacket(None, body), ValuePacket(None, wire)],
            message="Landmark body/wire frames conflict",
        )
        context = {**body, **wire}
        target = self.input_payloads.get("image", {})
        for key in ("ticket", "recipe_ref"):
            known = [ctx[key] for ctx in (body, wire, target) if key in ctx]
            keys = [ticket_key(v) for v in known] if key == "ticket" else known
            if keys and any(v != keys[0] for v in keys[1:]):
                raise ValidationError("Landmark pose inputs have conflicting " + key)
        chosen = []
        for index in (
            [p["origin_index"], p["direction_index"]]
            if p["angle_mode"] == "pair"
            else [p["origin_index"]]
        ):
            if index >= len(data["items"]):
                break
            item = data["items"][index]
            if (
                not isinstance(item, dict)
                or item.get("index") != index
                or type(item.get("accepted")) is not bool
            ):
                raise ValidationError("Landmarks must retain their declared ordered indices")
            if not item["accepted"]:
                break
            point = np.asarray(item.get("position"), float)
            if point.shape != (2,) or not np.isfinite(point).all() or np.max(np.abs(point)) > 1e7:
                raise ValidationError("Accepted landmark position must be finite XY")
            chosen.append(point)
        needed = 2 if p["angle_mode"] == "pair" else 1
        found = len(chosen) == needed
        angle = p["angle_degrees"]
        if found and needed == 2:
            delta = chosen[1] - chosen[0]
            found = np.linalg.norm(delta) >= p["minimum_baseline"]
            angle = -math.degrees(math.atan2(delta[1], delta[0]))
        pose = NO_VALUE
        if found:
            pose = make_pose(local_offset(*chosen[0], angle), data.get("image_size"), context)
            pose = pose_for_image(offset_pose(pose, *(p[k] for k in POSE_OFFSETS)), source)
        outputs = {"image": source, "pose": pose, "found": bool(found)}
        return Result(
            outputs,
            payloads={key: copy.deepcopy(target) for key in outputs},
            passed=bool(found),
            message="Landmark pose available"
            if found
            else "Required landmarks or baseline unavailable",
            fail_reasons=[]
            if found
            else [
                {
                    "code": "landmark.pose_unavailable",
                    "message": "Required accepted landmarks or stable baseline unavailable",
                }
            ],
        )


LANDMARK_BLOCKS = (DecodeLandmarks, LandmarkPose)
