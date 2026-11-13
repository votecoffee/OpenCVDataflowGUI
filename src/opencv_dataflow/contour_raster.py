"""Topology-aware binary masks from contour or blob collections."""

import copy

import cv2
import numpy as np

from .collection_tools import _local_collection_payload, read_collection
from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, frame, image_with_payload, require_same_frame
from .result_graphics import ACCEPTED, REJECTED
from .selection import selection_mask
from .tickets import ticket_key

LINE_TYPES = {"8_connected": cv2.LINE_8, "4_connected": cv2.LINE_4}


def contour_tree(ids, links):
    """Validate ID-based native links, including the completeness of each sibling chain."""
    message = (
        "Hierarchy is incomplete or inconsistent. Supply the unfiltered contour tree and "
        "select items here, or explicitly choose solid/even_odd topology."
    )
    if any(type(key) is not int or key < 0 for key in ids):
        raise ValidationError(message)
    keys = set(ids)
    rows = {}
    for key, row in zip(ids, links, strict=True):
        if (
            not isinstance(row, list)
            or len(row) != 4
            or any(type(value) is not int or value < -1 for value in row)
            or any(value != -1 and value not in keys for value in row)
        ):
            raise ValidationError(message)
        rows[key] = row
    children = {-1: []}
    children.update({key: [] for key in ids})
    for key, (following, previous, first_child, parent) in rows.items():
        children[parent].append(key)
        if following != -1 and (rows[following][1] != key or rows[following][3] != parent):
            raise ValidationError(message)
        if previous != -1 and (rows[previous][0] != key or rows[previous][3] != parent):
            raise ValidationError(message)
        if first_child != -1 and (rows[first_child][3] != key or rows[first_child][1] != -1):
            raise ValidationError(message)
    for parent, members in children.items():
        heads = [key for key in members if rows[key][1] == -1]
        if parent != -1 and heads != ([] if not members else [rows[parent][2]]):
            raise ValidationError(message)
        visited = set()
        for head in heads:
            current = head
            while current != -1:
                if current in visited:
                    raise ValidationError(message)
                visited.add(current)
                current = rows[current][0]
        if visited != set(members):
            raise ValidationError(message)
    depths = {}
    for key in ids:
        path, seen, current = [], set(), key
        while current != -1 and current not in depths:
            if current in seen:
                raise ValidationError(message)
            seen.add(current)
            path.append(current)
            current = rows[current][3]
        depth = depths.get(current, -1)
        for member in reversed(path):
            depth += 1
            depths[member] = depth
    return rows, children, depths


def selected_groups(ids, links, selected, include_descendants, maximum_depth, *, roots_only=False):
    rows, children, depths = contour_tree(ids, links)
    selected = set(children[-1] if roots_only else selected)
    anchors = []
    coverage = {}
    for key in sorted(ids, key=depths.get):
        remaining = max(-1, coverage.get(rows[key][3], -1) - 1)
        coverage[key] = remaining
        if key not in selected or (include_descendants and remaining >= 0):
            continue
        anchors.append(key)
        coverage[key] = (
            (len(ids) if maximum_depth < 0 else maximum_depth) if include_descendants else 0
        )
    groups = []
    for key in anchors:
        group, pending = [], [(key, 0)]
        while pending:
            member, depth = pending.pop()
            group.append(member)
            if include_descendants and (maximum_depth < 0 or depth < maximum_depth):
                pending.extend((child, depth + 1) for child in children[member])
        groups.append(group)
    return groups


def select_items(items, selection, requested):
    if selection == "all":
        return list(range(len(items)))
    if not isinstance(requested, list) or len(requested) > 100_000:
        raise ValidationError("Items must be a list of up to 100,000 indices or IDs", "items")
    indices = []
    lookup = {item["id"]: index for index, item in enumerate(items)}
    for key in requested:
        if selection == "indices":
            if type(key) is not int or not 0 <= key < len(items):
                raise ValidationError("Item indices must be valid zero-based integers", "items")
            indices.append(key)
        else:
            if type(key) not in (int, str) or key not in lookup:
                raise ValidationError(f"Unknown collection item ID: {key!r}", "items")
            indices.append(lookup[key])
    if len(set(indices)) != len(indices):
        raise ValidationError("Selected item indices or IDs must be unique", "items")
    return indices


def raster_geometry(data, chosen, parameters):
    items, topology = data["items"], parameters["topology"]
    points, groups = {}, []
    if data["kind"] == "contours":
        ids = [item["id"] for item in items]
        points = {item["id"]: item["geometry"].get("points") for item in items}
        selected = [items[index]["id"] for index in chosen]
        if topology == "hierarchy":
            groups = selected_groups(
                ids,
                [item["geometry"].get("hierarchy") for item in items],
                selected,
                parameters["include_descendants"],
                parameters["maximum_depth"],
                roots_only=parameters["selection"] == "all",
            )
        elif topology == "solid":
            groups = [[key] for key in selected]
        else:
            groups = [selected] if selected else []
    else:
        for index in chosen:
            geometry = items[index]["geometry"]
            contours = geometry.get("contours")
            if not isinstance(contours, list) or not 1 <= len(contours) <= 100_000:
                raise ValidationError("Blob geometry needs a nonempty contour list")
            ids = list(range(len(contours)))
            if topology == "hierarchy":
                links = geometry.get("hierarchy")
                if not isinstance(links, list) or len(links) != len(ids):
                    raise ValidationError("Blob geometry needs a complete contour hierarchy")
                local = selected_groups(
                    ids,
                    links,
                    ids,
                    parameters["include_descendants"],
                    parameters["maximum_depth"],
                    roots_only=True,
                )
            else:
                local = [[key] for key in ids] if topology == "solid" else [ids]
            points.update({(index, key): contour for key, contour in enumerate(contours)})
            groups.extend([[(index, key) for key in group] for group in local])
    return points, groups


class RasterizeContours(Block):
    type_id, title = "collection.rasterize_contours", "Rasterize contours with holes"
    category, block_type, subtype = "Image processing", "Image processing", "Regions"
    inputs = (
        Port("collection", "data"),
        Port("image", required=False, required_when_connected=True),
        Port("items", "data", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("mask"),
        Port("count", "number"),
        Port("pixel_count", "integer"),
        Port("fraction", "number"),
        Port("info", "data"),
        Port("image", required=False),
        Port("overlay", required=False),
    )
    defaults = {
        "selection": "all",
        "items": [],
        "topology": "hierarchy",
        "include_descendants": True,
        "maximum_depth": -1,
        "mode": "filled",
        "thickness": 1,
        "line_type": "8_connected",
        "subpixel_bits": 0,
        "offset": [0.0, 0.0],
        "invert": False,
        "alignment": "same_frame",
        "check_pixels": False,
        "minimum_pixels": 0,
        "maximum_pixels": 64_000_000,
    }
    parameter_choices = {
        "selection": ("all", "indices", "ids"),
        "topology": ("hierarchy", "solid", "even_odd"),
        "mode": ("filled", "outline"),
        "line_type": tuple(LINE_TYPES),
        "alignment": ("same_frame", "pixel_grid"),
    }
    parameter_ranges = {
        "maximum_depth": NumericRange(-1, 100_000, whole=True),
        "thickness": NumericRange(1, 128, whole=True),
        "subpixel_bits": NumericRange(0, 8, whole=True),
        "minimum_pixels": NumericRange(0, 64_000_000, whole=True),
        "maximum_pixels": NumericRange(0, 64_000_000, whole=True),
    }
    parameter_help = {
        "selection": "All items, zero-based positions in the current order, or stable item IDs.",
        "items": "Selected indices/IDs; a wired items input overrides this list. Ignored for all.",
        "topology": (
            "Hierarchy preserves selected subtrees; solid unions polygons; even_odd uses parity."
        ),
        "include_descendants": "Hierarchy only: include child holes and nested islands.",
        "maximum_depth": "Hierarchy only: -1 includes all descendants; 0 draws roots only.",
        "mode": "Fill interiors or draw closed outlines. Both emit binary uint8 masks.",
        "thickness": "Outline width in pixels. Ignored for filled masks.",
        "line_type": "Binary boundary connectivity. Antialiasing is excluded to keep masks binary.",
        "subpixel_bits": "0 rounds to whole pixels; 1–8 retain fixed-point fractional bits.",
        "offset": "Additional [x,y] shift in output pixels, after subtracting collection origin.",
        "invert": "Complement the raster before clipping to the supplied mask/region.",
        "alignment": "Match the image frame or use its pixel grid. Known tickets must match.",
        "check_pixels": "Judge the final selected foreground pixel count against inclusive limits.",
        "minimum_pixels": "Smallest accepted count after inversion and selection clipping.",
        "maximum_pixels": "Largest accepted count after inversion and selection clipping.",
    }
    input_help = {
        "collection": "Contour tree or blobs. Use the full tree and select here to retain holes.",
        "image": "Optional equal-sized frame reference and unchanged image output.",
        "items": "Optional replacement item indices/IDs when selection is not all.",
        "mask": "Optional uint8 selection in the output frame; nonzero pixels are eligible.",
        "region": "Optional region intersected with the mask; rasterize first, then clip.",
    }
    parallel_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["mask", "count", "pixel_count", "info"]
        self.documentation = (
            "<p>Rasterize each related contour subtree as a group, then union independent roots. "
            "This preserves holes and islands without XOR cancellation between independent parts. "
            "A selected hole is treated as a new foreground root. Missing hierarchy links are "
            "reported; no absent holes are invented.</p>"
            '<p><a href="https://docs.opencv.org/5.0/main_modules/imgproc_draw.html">'
            "OpenCV drawing reference</a>. Subpixel fills use fillPoly; whole-pixel fills use "
            "drawContours. Outlines use polylines. Masks are binary, not alpha coverage.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        try:
            offset = np.asarray(p["offset"], float)
        except (ValueError, TypeError) as exc:
            raise ValidationError(
                "Offset must be a finite [x,y] pair within ±1M pixels", "offset"
            ) from exc
        if (
            offset.shape != (2,)
            or not np.isfinite(offset).all()
            or np.any(np.abs(offset) > 1_000_000)
        ):
            raise ValidationError("Offset must be a finite [x,y] pair within ±1M pixels", "offset")
        if not isinstance(p["items"], list):
            raise ValidationError("Items must be a JSON list", "items")
        if p["minimum_pixels"] > p["maximum_pixels"]:
            raise ValidationError("Minimum pixels must not exceed maximum pixels", "minimum_pixels")

    def process(self, inputs):
        try:
            data, payload = read_collection(self, inputs)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValidationError(str(exc)) from exc
        if data["kind"] not in ("contours", "blobs"):
            raise ValidationError("Rasterization requires a contour or blob collection")
        p, payload = self.parameters, _local_collection_payload(data, payload)
        image = inputs.get("image")
        width, height = data["image_size"]
        if image is not None:
            if image.shape[:2] != (height, width):
                raise ValidationError(
                    "Reference image and collection must have matching dimensions"
                )
            image_payload = self.input_payloads["image"]
            if p["alignment"] == "same_frame":
                try:
                    require_same_frame(
                        (image, ValuePacket(None, payload)),
                        message="Reference image and collection use different coordinate frames",
                    )
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                source, target = payload.get("spatial"), image_payload.get("spatial")
                if (
                    source
                    and target
                    and (
                        source.get("root_id") == target.get("root_id")
                        and (
                            source["root_size"] != target["root_size"]
                            or not np.allclose(
                                source["to_root"],
                                target["to_root"],
                                atol=1e-9,
                                rtol=0,
                            )
                        )
                    )
                ):
                    raise ValidationError(
                        "Reference image and collection use different pixel grids"
                    )
            if (
                "ticket" in image_payload
                and "ticket" in payload
                and (ticket_key(image_payload["ticket"]) != ticket_key(payload["ticket"]))
            ):
                raise ValidationError("Reference image and collection have different tickets")
            known_collection_frame = payload.get("spatial", {}).get("root_id") is not None
            anonymous_image_frame = image_payload.get("spatial", {}).get("root_id") is None
            keep_collection_grid = (
                p["alignment"] == "same_frame" and known_collection_frame and anonymous_image_frame
            )
            retained_grid = {
                name: copy.deepcopy(payload[name])
                for name in ("spatial", "pyramid")
                if name in payload and keep_collection_grid
            }
            payload = {**payload, **copy.deepcopy(image_payload), **retained_grid}
        payload.setdefault("spatial", frame(data["image_size"]))
        chosen = select_items(data["items"], p["selection"], inputs.get("items", p["items"]))
        points, groups = raster_geometry(data, chosen, p)
        scale = 1 << p["subpixel_bits"]
        shift = np.asarray(p["offset"], float) - np.asarray(data.get("origin", [0, 0]), float)
        native, point_count, work = {}, 0, 0
        for group in groups:
            bounds = []
            for key in group:
                if key not in native:
                    try:
                        vertices = np.asarray(points[key], float)
                    except (ValueError, TypeError) as exc:
                        raise ValidationError(
                            "Contour vertices must be numeric [x,y] pairs"
                        ) from exc
                    if (
                        vertices.ndim != 2
                        or vertices.shape[1] != 2
                        or not 1 <= len(vertices) <= 100_000
                        or not np.isfinite(vertices).all()
                        or np.any(np.abs(vertices + shift) > 1_000_000)
                    ):
                        raise ValidationError(
                            "Contours need 1–100,000 finite [x,y] points within ±1M output pixels"
                        )
                    point_count += len(vertices)
                    if point_count > 1_000_000:
                        raise ValidationError(
                            "Rasterization is limited to one million selected vertices"
                        )
                    native[key] = np.rint((vertices + shift) * scale).astype(np.int32)
                bounds.append(native[key])
            vertices = np.concatenate(bounds).astype(float) / scale
            low = np.maximum(0, np.floor(vertices.min(axis=0)) - p["thickness"])
            high = np.minimum([width, height], np.ceil(vertices.max(axis=0)) + p["thickness"] + 1)
            work += float(np.prod(np.maximum(0, high - low)))
            if work > 256_000_000:
                raise ValidationError(
                    "Rasterization exceeds 256M bounded group pixels; split the collection"
                )
        mask = np.zeros((height, width), np.uint8)
        line = LINE_TYPES[p["line_type"]]
        for group in groups:
            contours = [native[key] for key in group]
            if p["mode"] == "outline":
                cv2.polylines(mask, contours, True, 255, p["thickness"], line, p["subpixel_bits"])
            elif p["subpixel_bits"]:
                cv2.fillPoly(mask, contours, 255, line, p["subpixel_bits"])
            else:
                cv2.drawContours(mask, contours, -1, 255, cv2.FILLED, line)
        if p["invert"]:
            cv2.bitwise_not(mask, mask)
        canvas = image_with_payload(mask, payload)
        selection = selection_mask({**inputs, "image": canvas}, canvas)
        if selection is not None:
            mask[selection == 0] = 0
        pixels = int(np.count_nonzero(mask))
        eligible = mask.size if selection is None else int(np.count_nonzero(selection))
        passed = not p["check_pixels"] or p["minimum_pixels"] <= pixels <= p["maximum_pixels"]
        info = {
            "model": "contour_raster_v1",
            "source_kind": data["kind"],
            "selected_ids": [data["items"][index]["id"] for index in chosen],
            "selected_indices": chosen,
            "drawn_contours": len(native),
            "groups": len(groups),
            "foreground_pixels": pixels,
            "eligible_pixels": eligible,
            "image_fraction": pixels / mask.size,
            "topology": p["topology"],
            "mode": p["mode"],
            "subpixel_bits": p["subpixel_bits"],
        }
        objects, overlay = [], NO_VALUE
        if self.generate_graphics:
            color, budget = ACCEPTED if passed else REJECTED, 20_000
            for vertices in native.values():
                if budget == 0 or len(objects) >= 500:
                    break
                stride = max(1, int(np.ceil(len(vertices) / min(budget, 1000))))
                reduced = vertices[::stride].astype(float) / scale
                objects.append(graphic_object("polyline", reduced, color, closed=True))
                budget -= len(reduced)
            objects.append(
                graphic_object("text", [[5, 16]], color, text=f"{pixels} foreground pixels")
            )
            overlay = overlay_image(image if image is not None else canvas)
            for obj in objects:
                draw_object(overlay, obj, np.eye(3))
        outputs = {
            "mask": canvas,
            "count": len(chosen),
            "pixel_count": pixels,
            "fraction": pixels / eligible if eligible else 0.0,
            "info": info,
            "image": image if image is not None else NO_VALUE,
            "overlay": overlay,
        }
        return Result(
            outputs,
            payloads={name: payload for name in outputs},
            passed=passed,
            graphics={"rasterized contours": overlay} if overlay is not NO_VALUE else {},
            drawings={"rasterized contours": GraphicDrawing(tuple(objects), "mask", True)}
            if objects
            else {},
            message=f"{len(chosen)} selected · {len(native)} boundaries · {pixels} pixels",
            fail_reasons=[] if passed else ["raster_pixel_count_out_of_range"],
        )
