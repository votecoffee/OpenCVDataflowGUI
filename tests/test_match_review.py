"""Verify feature match graphics stay on their comparison canvas."""

import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED
from opencv_dataflow.tickets import TicketClock


def views():
    query = np.full((32, 50), 25, np.uint8)
    train = np.full((40, 65, 3), 45, np.uint8)
    matches = {
        "query_points": [[12, 12], [15, 23]],
        "train_points": [[8, 12], [9, 23]],
        "query_size": [50, 32],
        "train_size": [65, 40],
    }
    return query, train, matches


def test_match_layers_render_and_compose_in_the_output_frame():
    query, train, matches = views()
    ticket = TicketClock().issue("query")
    packet = ValuePacket(query, {"spatial": frame([50, 32], "query"), "ticket": ticket})
    registry = builtin_registry()
    block = registry.create("inspect.review_matches", parameters={"endpoints": False})
    block._event_path = "inspection/review"
    block.exposed_outputs.append(RESULT_GRAPHICS)
    result = block.execute({
        "query_image": packet, "train_image": train, "matches": matches,
        "fit": {"inlier_mask": [True, False]},
    })
    assert result is not None, block.error
    canvas, overlay = result.outputs["canvas"], result.outputs["overlay"]
    assert canvas.shape == overlay.shape == (40, 131, 3)
    assert result.outputs["info"]["inliers"] == result.outputs["info"]["outliers"] == 1
    assert tuple(overlay[12, 57]) == ACCEPTED
    assert tuple(overlay[23, 57]) == REJECTED
    assert tuple(canvas[23, 57]) == (30, 30, 30)
    context = result.payloads["canvas"]
    assert context["spatial"]["root_id"] == "inspection/review:matches"
    assert context["ticket"] == ticket
    layers = result.outputs[RESULT_GRAPHICS].layers
    assert {layer.name for layer in layers} == {"Inlier matches", "Outlier matches"}
    for layer in layers:
        assert layer.payload["spatial"] == context["spatial"]
    compose = registry.create("cv.overlay_result_graphics", parameters={
        "excluded_graphics": {"graphics_1": ["Outlier matches"]},
    })
    shown = compose.execute({
        "image": packet_for(result, "canvas"),
        "graphics_1": packet_for(result, RESULT_GRAPHICS),
    })
    assert shown is not None, compose.error
    assert tuple(shown.outputs["image"][12, 57]) == ACCEPTED
    assert tuple(shown.outputs["image"][23, 57]) == (30, 30, 30)


def test_match_review_filters_before_applying_display_cap():
    query, train, matches = views()
    block = builtin_registry().create("inspect.review_matches", parameters={
        "show": "outliers", "max_pairs": 1,
    })
    result = block.execute({
        "query_image": query, "train_image": train, "matches": matches,
        "fit": {"inlier_mask": [True, False]},
    })
    assert result is not None, block.error
    assert set(result.drawings) == {"Outlier matches"}
    assert result.outputs["info"]["shown"] == 1


def test_match_review_rejects_mismatched_fit_and_image_size():
    query, train, matches = views()
    block = builtin_registry().create("inspect.review_matches")
    inputs = {"query_image": query, "train_image": train, "matches": matches}
    assert block.execute({**inputs, "fit": {"inlier_mask": [True]}}) is None
    assert "one boolean" in block.error
    assert block.execute({**inputs, "train_image": train[:, :-1]}) is None
    assert "image size differs" in block.error
    empty = {**matches, "query_points": [], "train_points": []}
    result = block.execute({**inputs, "matches": empty})
    assert result is not None, block.error
    assert result.outputs["info"]["shown"] == 0
    np.testing.assert_array_equal(result.outputs["canvas"], result.outputs["overlay"])
