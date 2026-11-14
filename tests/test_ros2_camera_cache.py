"""Static calibration reuse, explicit binding provenance and run/reset isolation."""

import copy

import numpy as np
import pytest
from test_ros2_calibration import model

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, metadata
from opencv_dataflow.ros2_calibration import AssociateROSRGBD, camera_from_info, camera_to_info
from opencv_dataflow.ros2_camera_cache import CacheROSCamera, calibration_revision
from opencv_dataflow.ros2_context import AttachAcquisition, RunIdentity
from opencv_dataflow.ros2_images import decode_image, encode_image
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry


def calibration():
    return camera_from_info(camera_to_info(model()))[0]


def image(number=1, stamp=18):
    header = {"frame_id": "camera_optical", "stamp": {"sec": stamp, "nanosec": number}}
    value = decode_image(
        encode_image(
            ImageArray(np.full((4, 6), 1000, np.uint16), "gray", {"ros2": {"header": header}})
        )
    )
    return ValuePacket(value, {"ticket": TicketClock("cache-test").issue("depth", "fixed", number)})


def execute(block, **inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def bound_group(result):
    return FlowGroup(
        tuple(ValuePacket(result.outputs[k], result.payloads[k]) for k in ("image", "camera")),
        ("depth", "camera"),
    )


def rgbd(**parameters):
    return AssociateROSRGBD(
        parameters={
            "use_color": False,
            "depth_label": "depth",
            "camera_label": "camera",
            **parameters,
        }
    )


def test_static_update_emits_nothing_then_binds_many_images_without_rewriting_calibration():
    cache, original = CacheROSCamera(), calibration()
    assert execute(cache, camera=original).outputs["camera"] is NO_VALUE
    assert not any(cache.current_emissions.values())
    for number in (1, 2):
        incoming = image(number)
        result = execute(cache, image=incoming)
        bound = result.outputs["camera"]
        assert bound["ros2"] == original["ros2"]
        assert bound["ros2_static_binding"]["image_header"]["stamp"]["sec"] == 18
        assert result.outputs["info"]["calibration_header"]["stamp"]["sec"] == 17
        assert ticket_key(result.payloads["camera"]["ticket"]) == ticket_key(
            incoming.payload["ticket"]
        )
        np.testing.assert_array_equal(result.outputs["image"], incoming.value)
    result.outputs["camera"]["camera_matrix"][0][0] = 999
    assert cache.runtime["cached"]["camera"]["camera_matrix"][0][0] == 10
    assert "ros2_static_binding" not in original


def test_images_before_calibration_are_dropped_and_never_replayed():
    cache = CacheROSCamera()
    assert execute(cache, image=image()).outputs["image"] is NO_VALUE
    assert execute(cache, camera=calibration()).outputs["image"] is NO_VALUE
    assert not any(cache.current_emissions.values())
    assert execute(cache, image=image(2)).outputs["info"]["image_ticket"]["value"] == 2


def test_receipt_expiry_uses_monotonic_time_and_does_not_refresh_on_image_events():
    cache, clock = CacheROSCamera(parameters={"max_age_seconds": 2}), [100.0]
    cache.now = lambda: clock[0]
    execute(cache, camera=calibration())
    clock[0] += 1.5
    assert execute(cache, image=image()).outputs["info"]["receipt_age_seconds"] == 1.5
    clock[0] += 0.6
    assert execute(cache, image=image(2)).outputs["image"] is NO_VALUE
    assert cache.runtime["cached"] is None
    execute(cache, camera=calibration())
    assert execute(cache, image=image(3)).outputs["image"] is not NO_VALUE


def test_revision_ignores_stamp_and_field_order_but_tracks_actual_calibration():
    first = calibration()
    second = copy.deepcopy(first)
    second["ros2"]["header"]["stamp"]["sec"] = 19
    second["ros2"]["camera_info"]["header"]["stamp"]["sec"] = 19
    second["ros2"]["camera_info"] = dict(reversed(list(second["ros2"]["camera_info"].items())))
    assert calibration_revision(first) == calibration_revision(second)
    changed = copy.deepcopy(second["ros2"]["camera_info"])
    changed["k"][0] = 20.0
    third = camera_from_info(changed)[0]
    assert calibration_revision(first) != calibration_revision(third)
    cache = CacheROSCamera()
    execute(cache, camera=first)
    before = execute(cache, image=image()).outputs["info"]["calibration_revision"]
    execute(cache, camera=third)
    assert execute(cache, image=image(2)).outputs["info"]["calibration_revision"] != before


@pytest.mark.parametrize("change", ["frame", "size", "spatial", "stamp", "ticket", "ros_grid"])
def test_incompatible_image_is_contained_and_invalidates_cache(change):
    cache, value = CacheROSCamera(), image()
    execute(cache, camera=calibration())
    if change == "frame":
        value.value.metadata["ros2"]["header"]["frame_id"] = "other_camera"
    elif change == "size":
        value.value = value.value[:2].copy()
    elif change == "spatial":
        value.value.metadata["spatial"] = {
            "image_size": [6, 4],
            "root_id": "crop",
            "to_root": [[1, 0, 1], [0, 1, 0], [0, 0, 1]],
        }
    elif change == "stamp":
        value.value.metadata["ros2"]["header"]["stamp"] = {"sec": 0, "nanosec": 0}
    elif change == "ticket":
        value.payload.clear()
    else:
        value.value.metadata["ros2"]["width"] = 99
    assert cache.execute({"image": value}) is None
    assert not cache.current_emissions and cache.runtime["cached"] is None


def test_zero_stamp_is_an_explicit_offline_option():
    cache = CacheROSCamera(parameters={"allow_zero_stamps": True})
    execute(cache, camera=calibration())
    incoming = image(0, stamp=0)
    result = execute(cache, image=incoming)
    checked = rgbd(calibration_timing="bound_static", allow_zero_stamps=True)
    result = execute(checked, group=bound_group(result))
    assert not result.outputs["info"]["timestamps_nonzero"]
    assert cache.execute({"image": image(stamp=-1)}) is None


def test_bound_static_is_explicit_and_preserves_independent_depth_projection():
    cache = CacheROSCamera()
    execute(cache, camera=calibration())
    group = bound_group(execute(cache, image=image()))
    assert rgbd().execute({"group": group}) is None  # Original CameraInfo is older.
    checked = execute(rgbd(calibration_timing="bound_static"), group=group)
    assert checked.outputs["info"]["header_spread_ns"] == 0
    assert checked.outputs["info"]["static_calibration"]["calibration_header"]["stamp"]["sec"] == 17
    np.testing.assert_allclose(checked.outputs["depth"], 1.0)
    cloud = builtin_registry().create("3d.depth_cloud", parameters={"sample_step": 1})
    result = execute(cloud, depth=checked.outputs["depth"], camera=checked.outputs["camera"])
    np.testing.assert_allclose(result.outputs["cloud"].points[0], [-0.2, -1 / 12, 1], rtol=1e-6)


@pytest.mark.parametrize(
    "change", ["revision", "header", "size", "ticket", "scope", "age", "geometry", "image_header"]
)
def test_bound_static_does_not_bypass_binding_validation(change):
    cache = CacheROSCamera()
    execute(cache, camera=calibration())
    group = bound_group(execute(cache, image=image()))
    camera = group.items[1].value
    binding = camera["ros2_static_binding"]
    if change == "revision":
        binding["calibration_revision"] = "old"
    elif change == "header":
        binding["calibration_header"]["stamp"]["sec"] = 100
    elif change == "size":
        binding["image_size"] = [1, 1]
    elif change == "ticket":
        binding["image_ticket"]["value"] = "other"
    elif change == "scope":
        binding["context_scope"] = "old-run"
    elif change == "age":
        binding["receipt_age_seconds"] = -1
    elif change == "geometry":
        camera["camera_matrix"][0][0] = 123
    else:
        binding["image_header"]["stamp"]["sec"] += 1
    block = rgbd(calibration_timing="bound_static")
    assert block.execute({"group": group}) is None
    assert not block.current_emissions


def test_context_change_and_reset_require_repriming_and_reject_stale_acquisitions():
    run, cache = RunIdentity(), CacheROSCamera(parameters={"require_context": True})
    current = execute(run).outputs["context"]
    assert execute(cache, camera=calibration()).outputs["image"] is NO_VALUE
    execute(cache, context=current)
    execute(cache, camera=calibration())
    attach = AttachAcquisition()

    def tagged(context, number=1):
        result = execute(attach, value=image(number), context=context)
        return ValuePacket(result.outputs["value"], result.payloads["value"])

    old = tagged(current)
    assert execute(cache, image=old).outputs["info"]["context"] == current
    new = execute(run, reset=True).outputs["context"]
    execute(cache, context=new)
    assert execute(cache, image=tagged(new, 2)).outputs["image"] is NO_VALUE
    execute(cache, camera=calibration())
    assert cache.execute({"image": old}) is None
    assert cache.runtime["cached"] is None
    execute(cache, context=new)
    execute(cache, camera=calibration())
    execute(cache, reset=True)
    assert cache.runtime["cached"] is None
    execute(cache, camera=calibration())
    execute(cache, reset=True)
    assert execute(cache, image=tagged(new, 3)).outputs["image"] is not NO_VALUE


def test_copy_runtime_is_private_and_project_xml_contains_no_cached_samples(tmp_path):
    cache, other = CacheROSCamera(), CacheROSCamera()
    execute(cache, camera=calibration())
    other.copy_runtime_from(cache)
    other.runtime["cached"]["camera"]["frame"] = "changed"
    assert cache.runtime["cached"]["camera"]["frame"] == "camera_optical"
    graph = Graph()
    graph.add(cache)
    path = tmp_path / "cache.xml"
    graph.save(path)
    assert "ros_static_camera_binding_v1" not in path.read_text()
    restored = Graph.load(path, builtin_registry())
    loaded = next(iter(restored.blocks.values()))
    assert loaded.runtime["cached"] is None and loaded.preview_safe is False
    assert block_entry(loaded).path == ("Industrial integration", "ROS 2")
    restored.close()
    graph.close()


def test_failed_or_previously_bound_calibration_cannot_replace_the_cache():
    cache = CacheROSCamera()
    execute(cache, camera=calibration())
    result = execute(cache, image=image())
    assert cache.execute({"camera": result.outputs["camera"]}) is None
    assert cache.runtime["cached"] is None
    execute(cache, camera=calibration())
    failed = ValuePacket(
        calibration(),
        {
            "outcome": {
                "model": "tool_outcome_v1",
                "passed": False,
                "chain_passed": False,
                "source": {"id": "camera"},
                "score": None,
                "message": "bad",
                "data": {},
            }
        },
    )
    assert cache.execute({"camera": failed}) is None
    assert cache.runtime["cached"] is None


def test_cached_managed_recipe_is_preserved_and_inconsistent_revisions_are_rejected():
    cache = CacheROSCamera()
    ref = {"id": "inspection", "revision": 1, "digest": "a" * 64}
    execute(cache, camera=ValuePacket(calibration(), {"recipe_ref": ref}))
    result = execute(cache, image=image())
    assert result.payloads["camera"]["recipe_ref"] == ref
    assert execute(rgbd(calibration_timing="bound_static"), group=bound_group(result))
    value = image(2)
    value.payload["recipe_ref"] = {**ref, "revision": 2}
    assert cache.execute({"image": value}) is None
    assert cache.runtime["cached"] is None


@pytest.mark.parametrize(
    "parameters", [{"max_age_seconds": -1}, {"max_age_seconds": True}, {"require_context": "yes"}]
)
def test_parameter_validation_is_contained(parameters):
    cache = CacheROSCamera(parameters=parameters)
    assert cache.execute({"camera": calibration()}) is None
    assert cache.error_kind == "invalid"


def test_graph_event_order_never_replays_an_image_on_a_calibration_update():
    class Source(Block):
        outputs = (Port("camera", "data"), Port("image"))

        def process(self, inputs):
            value = image()
            return Result(
                {"camera": calibration(), "image": value.value}, payloads={"image": metadata(value)}
            )

    graph = Graph()
    source = graph.add(Source())
    cache = graph.add(CacheROSCamera())
    graph.connect(source.id, "camera", cache.id, "camera")
    graph.connect(source.id, "image", cache.id, "image")
    graph.run()
    assert cache.last_result is not None, {b.id: b.error for b in graph.blocks.values()}
    assert len(cache.current_emissions["image"]) == 1
    assert cache.last_result.outputs["info"]["image_ticket"]["value"] == 1
    graph.close()


def test_static_camera_controls_and_rgbd_mode_are_available_offscreen(app):
    from PySide6.QtWidgets import QCheckBox, QComboBox

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        graph = Graph()
        cache = graph.add(CacheROSCamera(block_id="cache"))
        checked = graph.add(rgbd(calibration_timing="bound_static"))
        window.replace_graph(graph)
        window.scene.nodes[cache.id].setSelected(True)
        assert isinstance(window.inspector.fields["require_context"], QCheckBox)
        assert not window.inspector.can_preview()
        window.scene.clearSelection()
        window.scene.nodes[checked.id].setSelected(True)
        assert isinstance(window.inspector.fields["calibration_timing"], QComboBox)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
