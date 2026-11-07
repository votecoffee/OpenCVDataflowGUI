import copy
import json

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QInputDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.recipe_editor import RecipeManager
from opencv_dataflow.recipes import RecipeCatalog, digest, reference, validate_recipe
from opencv_dataflow.toolblocks import Binding, ToolBlock


@pytest.fixture
def assets():
    registry = builtin_registry()
    demo = registry.create("input.pattern_demo")
    demo.execute({})
    images = demo.last_result.outputs
    shape = registry.create("pattern.match_shape")
    texture = registry.create("pattern.match_texture")
    return (
        images,
        shape.train({"image": images["shape_reference"]}),
        texture.train({"image": images["texture_reference"]}),
    )


def add_recipes(graph, models=None):
    a = graph.recipes.put("A", {"threshold": 70, "border": "replicate", "min_score": 0.7}, models)
    b = graph.recipes.put("B", {"threshold": 180, "border": "reflect", "min_score": 0.8}, models)
    graph.recipes.activate(reference(a), project_default=True)
    return a, b


def pinned(image, recipe):
    return ValuePacket(image, {"recipe_ref": reference(recipe), "custom": {"part": 4}})


def test_catalog_revision_integrity_private_copies_and_runtime_default_distinction(assets):
    _, shape, texture = assets
    catalog = RecipeCatalog()
    a = catalog.put(1, {"threshold": 80}, {"part": shape, "surface": texture})
    catalog.activate(reference(a), project_default=True)
    a2 = catalog.put("1", {"threshold": 90}, {"part": shape})
    assert a2["revision"] == 2 and a2["digest"] != a["digest"]
    assert catalog.resolve(1) == a2
    assert catalog.resolve(reference(a)) == a
    catalog.resolve(1)["settings"]["threshold"] = -1
    assert catalog.resolve(1)["settings"]["threshold"] == 90
    catalog.activate(reference(a2))
    assert RecipeCatalog.from_dict(catalog.definition()).active_ref() == reference(a)
    copied = RecipeCatalog.from_dict(catalog.definition())
    copied.copy_runtime_from(catalog)
    assert copied.active_ref() == reference(a2)
    tampered = copy.deepcopy(a)
    tampered["settings"]["threshold"] = 300
    with pytest.raises(ValueError, match="checksum"):
        validate_recipe(tampered)
    conflict = copy.deepcopy(a)
    conflict["settings"]["threshold"] = 90
    conflict["digest"] = digest(conflict)
    incoming = RecipeCatalog.from_dict(
        {"model": "recipe_catalog_v1", "recipes": [conflict], "default": None}
    )
    before = catalog.definition()
    with pytest.raises(ValueError, match="already exists"):
        catalog.merge(incoming)
    assert catalog.definition() == before


@pytest.mark.parametrize(
    "settings, models",
    [([], {}), ({}, []), ({"bad": float("nan")}, {}), ({}, {"part": {"model": "fake"}})],
)
def test_invalid_recipe_content_does_not_modify_catalog(settings, models):
    catalog = RecipeCatalog()
    with pytest.raises((ValueError, TypeError)):
        catalog.put("bad", settings, models)
    assert catalog.definition()["recipes"] == []


def test_selection_fields_global_activation_and_pinned_revision_rules():
    registry = builtin_registry()
    graph = Graph()
    a, b = add_recipes(graph)
    select = graph.add(registry.create("recipe.select", parameters={"activate_global": True}))
    field = graph.add(registry.create("recipe.field"))
    graph.bind_recipes()
    image = np.zeros((12, 16), np.uint8)
    select.execute({"recipe": "B", "value": image})
    assert select.state == "pass", select.error
    assert graph.recipes.active_ref() == reference(b)
    assert graph.recipes.definition()["default"] == reference(a)
    field.execute({"recipe": select.current_emissions["recipe"][0]})
    assert field.last_result.outputs["value"] == 180
    select.execute({"value": pinned(image, a)})
    assert select.last_result.outputs["settings"]["threshold"] == 70
    assert graph.recipes.active_ref() == reference(b)  # Reading old work never reactivates A.
    select.execute({"recipe": a, "value": pinned(image, a)})
    assert select.state == "pass" and graph.recipes.active_ref() == reference(b)
    select.execute({"recipe": "B", "value": pinned(image, a)})
    assert select.state == "error" and not select.current_emissions
    select.parameters["rebind"] = True
    select.execute({"recipe": "B", "value": pinned(image, a)})
    assert select.last_result.payloads["value"]["recipe_ref"] == reference(b)
    field.parameters.update(path="settings.missing", missing="inactive")
    field.execute({"value": pinned(image, a)})
    assert field.state == "waiting" and field.last_result.outputs["value"] is NO_VALUE


def test_recipe_parameter_overrides_are_temporary_and_retain_effective_provenance():
    registry = builtin_registry()
    graph = Graph()
    a, b = add_recipes(graph)
    source = graph.add(registry.create("input.demo"))
    threshold = graph.add(registry.create("cv.threshold"))
    threshold.recipe_bindings = {"value": "settings.threshold"}
    graph.connect(source.id, "image", threshold.id, "image")
    graph.run()
    assert threshold.last_result.outputs["threshold"] == 70
    assert threshold.parameters["value"] == 127
    assert threshold.last_recipe_parameters == {"value": 70}
    assert threshold.last_recipe_ref == reference(a)
    row = graph.ticket_results.find(source.last_result.outputs["ticket"])
    assert row["recipes"] == [reference(a)] and row["events"][-1]["recipe"] == reference(a)
    graph.recipes.activate(reference(b))
    graph.run()
    assert threshold.last_result.outputs["threshold"] == 180
    assert threshold.parameters["value"] == 127
    threshold.recipe_bindings = {"value": "settings.missing"}
    graph.run()
    assert threshold.state == "error" and threshold.parameters["value"] == 127


def test_queue_drains_with_captured_recipe_after_active_and_same_id_revision_changes():
    registry = builtin_registry()
    graph = Graph()
    a, b = add_recipes(graph)
    source = graph.add(registry.create("input.demo"))
    queue = graph.add(registry.create("flow.image_queue"))
    threshold = graph.add(registry.create("cv.threshold"))
    threshold.recipe_bindings = {"value": "settings.threshold"}
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", threshold.id, "image")
    graph.bind_recipes()
    source.execute({})
    queue.enqueue(source.current_emissions["image"][0])
    a2 = graph.recipes.put("A", {"threshold": 240})
    graph.recipes.activate(reference(b))
    graph.run()
    assert threshold.last_result.outputs["threshold"] == 70
    assert threshold.last_recipe_ref == reference(a)
    graph.run()
    assert threshold.last_result.outputs["threshold"] == 180
    graph.recipes.activate(reference(a2))
    graph.run()
    assert threshold.last_result.outputs["threshold"] == 240


def test_activation_control_path_precedes_capture_and_shares_the_version():
    registry = builtin_registry()
    graph = Graph()
    _, b = add_recipes(graph)
    counter = graph.add(registry.create("input.counter"))
    select = graph.add(
        registry.create("recipe.select", parameters={"recipe_id": "B", "activate_global": True})
    )
    source = graph.add(registry.create("input.demo"))
    source.ticket_settings = {"mode": "external", "value": 1}
    threshold = graph.add(registry.create("cv.threshold"))
    threshold.recipe_bindings = {"value": "settings.threshold"}
    split = graph.add(registry.create("flow.split", parameters={"workers": 2}))
    graph.connect(counter.id, "value", select.id, "value")
    graph.connect(select.id, "value", source.id, "ticket")
    graph.connect(source.id, "image", threshold.id, "image")
    graph.connect(counter.id, "value", split.id, "value")
    graph.run()
    assert all(block.state == "pass" for block in graph.blocks.values())
    assert graph.recipes.active_ref() == reference(b)
    assert source.last_result.payloads["image"]["recipe_ref"] == reference(b)
    assert threshold.last_result.outputs["threshold"] == 180


@pytest.mark.parametrize(
    "kind, asset", [("pattern.match_shape", "shape"), ("pattern.match_texture", "texture")]
)
def test_model_swaps_via_binding_and_wired_model_and_mismatch_rejection(assets, kind, asset):
    images, shape, texture = assets
    registry = builtin_registry()
    graph = Graph()
    model = shape if asset == "shape" else texture
    a, b = add_recipes(graph, {"part": model})
    matcher = graph.add(
        registry.create(
            kind,
            parameters={"rotation_min": -20.0, "rotation_max": 20.0, "min_score": 0.7}
            if asset == "shape"
            else None,
        )
    )
    matcher.recipe_model = "models.part"
    parameter = "min_score" if asset == "shape" else "min_inlier_ratio"
    matcher.recipe_bindings = {parameter: "settings.min_score"}
    trainer = registry.create(kind)
    other = trainer.train({"image": cv2.flip(images[f"{asset}_reference"], 1)})
    b = graph.recipes.put("B", b["settings"], {"part": other})
    assert model["id"] != other["id"]
    graph.bind_recipes()
    graph.recipes.activate(reference(b))
    matcher.execute({"image": pinned(images["image"], a)})
    assert matcher.state == "pass", matcher.error
    assert matcher.trained_model is None and matcher.last_recipe_ref == reference(a)
    assert matcher.last_result.outputs["matches"]["pattern_id"] == model["id"]
    matcher.execute({"image": pinned(images["image"], b)})
    assert matcher.state in ("pass", "fail"), matcher.error
    assert matcher.last_result.outputs["matches"]["pattern_id"] == other["id"]
    field = graph.add(registry.create("recipe.field", parameters={"path": "models.part"}))
    field.execute({"value": pinned(images["image"], a)})
    matcher.recipe_model, matcher.recipe_bindings = "", {}
    matcher.execute(
        {"image": pinned(images["image"], a), "model": field.current_emissions["value"][0]}
    )
    assert matcher.state == "pass", matcher.error
    field.execute({"recipe": "B"})
    matcher.execute(
        {"image": pinned(images["image"], a), "model": field.current_emissions["value"][0]}
    )
    assert matcher.state == "error" and "different recipe" in matcher.error


def test_portable_nested_catalog_and_bindings_without_project_duplication(tmp_path):
    registry = builtin_registry()
    inner = Graph()
    a, _ = add_recipes(inner)
    threshold = inner.add(registry.create("cv.threshold"))
    threshold.recipe_bindings = {"value": "settings.threshold"}
    composite = ToolBlock(
        inner, {"image": Binding(threshold.id, "image")}, {"image": Binding(threshold.id, "image")}
    )
    graph = Graph()
    graph.add(composite)
    serialized = graph.to_dict()
    assert (
        "recipes" in serialized and "recipes" not in serialized["blocks"][0]["toolblock"]["graph"]
    )
    portable = tmp_path / "tool.json"
    composite.export(portable)
    restored = ToolBlock.read(portable, registry)
    restored.execute({"image": pinned(np.zeros((8, 12), np.uint8), a)})
    assert restored.state == "pass", restored.error
    project = tmp_path / "project.json"
    graph.save(project)
    assert Graph.load(project, registry).to_dict() == graph.to_dict()


def test_manager_import_copy_revision_and_inspector_bindings(app, monkeypatch, assets, tmp_path):
    window = MainWindow()
    manager = RecipeManager(window)
    manager.name.setText("Widget")
    manager.settings.setPlainText('{"threshold": 100}')
    first = manager.save_revision()
    assert first and manager.catalog.active_ref() == reference(first)
    manager.settings.setPlainText('{"threshold": 180}')
    second = manager.save_revision()
    assert second["revision"] == 2
    assert manager.catalog.active_ref() == reference(first)
    manager.activate()
    assert manager.catalog.active_ref() == reference(second)
    threshold = window.graph.add(window.registry.create("cv.threshold"))
    window.inspector.select(threshold)
    window.inspector.recipe_group.setChecked(True)
    window.inspector.recipe_fields["value"].setText("settings.threshold")
    assert window.inspector.apply()
    assert threshold.recipe_bindings == {"value": "settings.threshold"}
    restored = Graph.from_dict(window.graph.to_dict(), window.registry)
    assert restored.blocks[threshold.id].recipe_bindings == threshold.recipe_bindings
    matcher = window.registry.create("pattern.match_shape")
    matcher.trained_model = copy.deepcopy(assets[1])
    window.inspector.select(matcher)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("part", True))
    manager.copy_model()
    assert manager.models["part"]["id"] == assets[1]["id"]
    third = manager.save_revision()
    assert third["revision"] == 3 and third["models"]["part"]["id"] == assets[1]["id"]
    from PySide6.QtWidgets import QFileDialog

    path = tmp_path / "recipe.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    manager.export_recipe()
    original = path.read_bytes()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    manager.export_recipe()
    assert warnings and path.read_bytes() == original
    manager.close()
    window.dirty = False
    window.close()


def test_legacy_arbitrary_recipe_metadata_remains_valid_and_unknown_revision_fails():
    registry = builtin_registry()
    graph = Graph.load(
        "examples/opencv tools/channels_pyramids/channels_pyramids.json", registry
    )
    graph.run()
    assert all(b.state == "pass" for b in graph.blocks.values())
    a, _ = add_recipes(graph)
    ref = reference(a)
    ref["digest"] = "0" * 64
    field = graph.add(registry.create("recipe.field"))
    field.execute({"value": ValuePacket(np.zeros((8, 12), np.uint8), {"recipe_ref": ref})})
    assert field.state == "error" and "not installed" in field.error


def test_flow_recipe_bindings_rejected_and_bad_enum_override_restores_defaults():
    registry = builtin_registry()
    graph = Graph()
    recipe = graph.recipes.put("bad-border", {"border": "not-a-border"})
    graph.recipes.activate(reference(recipe))
    blur = graph.add(registry.create("cv.gaussian_blur"))
    blur.recipe_bindings = {"border": "settings.border"}
    blur.execute({"image": np.zeros((8, 12), np.uint8)})
    assert blur.state == "error" and blur.parameters["border"] == "reflect"
    data = graph.to_dict()
    data["blocks"][0]["recipe_bindings"] = {"unknown": "settings.border"}
    with pytest.raises(ValueError, match="recipe parameter"):
        Graph.from_dict(data, registry)
    queue = graph.add(registry.create("flow.image_queue"))
    queue.recipe_bindings = {"capacity": "settings.capacity"}
    queue.execute({})
    assert queue.state == "error" and "scheduling" in queue.error


def test_background_capture_keeps_invocation_recipe_when_global_switches_mid_capture():
    import threading

    from opencv_dataflow.core import Block, Port, Result
    from opencv_dataflow.queued_execution import QueuedExecution

    started, release = threading.Event(), threading.Event()

    class Source(Block):
        type_id = "input.recipe_test"
        outputs = (Port("image"),)

        def process(self, inputs):
            started.set()
            assert release.wait(3)
            return Result({"image": np.full((12, 16), 100, np.uint8)})

    registry = builtin_registry()
    registry.register(Source)
    graph = Graph()
    a, b = add_recipes(graph)
    source = graph.add(registry.create(Source.type_id))
    queue = graph.add(
        registry.create(
            "flow.image_queue", parameters={"mode": "background_acquisition", "capacity": 2}
        )
    )
    threshold = graph.add(registry.create("cv.threshold"))
    threshold.recipe_bindings = {"value": "settings.threshold"}
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", threshold.id, "image")
    execution = QueuedExecution(graph, registry)
    execution.start()
    try:
        assert started.wait(3)
        graph.recipes.activate(reference(b))
        release.set()
        assert execution.run_one()
        assert threshold.last_recipe_ref == reference(a)
        assert threshold.last_result.outputs["threshold"] == 70
        assert execution.run_one()
        assert threshold.last_recipe_ref == reference(b)
        assert threshold.last_result.outputs["threshold"] == 180
    finally:
        release.set()
        execution.close()


def test_parallel_parts_share_catalog_and_cannot_change_global_recipe():
    from opencv_dataflow.examples import parallel_parts_graph

    registry = builtin_registry()
    graph = parallel_parts_graph(registry)
    a = graph.recipes.put("A", {"min_area": 3})
    b = graph.recipes.put("B", {"min_area": 100000})
    graph.recipes.activate(reference(a), project_default=True)
    inspection = graph.blocks["parts_inspect"]
    child = inspection.graph.blocks["part_blobs"]
    child.recipe_bindings = {"min_area": "settings.min_area"}
    graph.run()
    assert inspection.last_result.outputs["pass_count"] == 25
    graph.recipes.activate(reference(b))
    graph.run()
    assert inspection.last_result.outputs["fail_count"] == 25
    preview = inspection.graph.blocks["part_blobs"]
    assert preview.parameters["min_area"] == 3
    assert preview.last_recipe_parameters["min_area"] == 100000
    inspection.graph.add(
        registry.create("recipe.select", parameters={"recipe_id": "A", "activate_global": True})
    )
    graph.run()
    assert inspection.last_result.outputs["error_count"] == 25
    assert graph.recipes.active_ref() == reference(b)
    assert any(
        "not allowed" in record["error"]
        for record in inspection.last_result.outputs["results"]["items"]
    )


def test_catalog_count_limit_preserves_prior_versions():
    catalog = RecipeCatalog()
    first = catalog.put("part", {"value": 0})
    for index in range(127):
        catalog.put("part", {"value": index + 1})
    with pytest.raises(ValueError, match="128 revisions"):
        catalog.put("part", {"value": 128})
    assert len(catalog.definition()["recipes"]) == 128
    assert catalog.resolve(reference(first)) == first


def test_queue_byte_budget_includes_captured_recipe_overrides():
    from opencv_dataflow.core import Result
    from opencv_dataflow.image_queue import QueuedCapture

    image = np.zeros((8, 12), np.uint8)
    snapshot = {"last_inputs": {}, "last_graphics": {}, "last_result": Result({"image": image})}
    capture = QueuedCapture(ValuePacket(image, {}), {"source": snapshot})
    before = capture.size()
    snapshot["last_recipe_parameters"] = {"data": "x" * 1000}
    assert capture.size() == before + 1000


@pytest.mark.parametrize(
    "kind, names",
    [
        ("flow.join", ("branch_1", "branch_2")),
        ("flow.coordinate", ("input_1", "input_2")),
        ("cv.blend", ("image_a", "image_b")),
    ],
)
def test_strict_processing_and_ticket_barriers_reject_mixed_recipe_revisions(kind, names):
    from opencv_dataflow.tickets import TicketClock

    graph = Graph()
    a, b = add_recipes(graph)
    block = graph.add(builtin_registry().create(kind))
    ticket = TicketClock().issue("camera", "fixed", "part")
    image = np.zeros((12, 16), np.uint8)
    inputs = {
        name: ValuePacket(image, {"ticket": ticket, "recipe_ref": reference(recipe)})
        for name, recipe in zip(names, (a, b))
    }
    block.execute(inputs)
    assert block.state == "error" and "different recipe revisions" in block.error
    assert not block.current_emissions


@pytest.mark.parametrize(
    "kind, name, value",
    [
        ("cv.canny", "l2_gradient", "false"),
        ("cv.gaussian_blur", "kernel", 3.5),
        ("cv.gaussian_blur", "sigma", "1.0"),
        ("cv.gaussian_blur", "border_value", 0),
    ],
)
def test_recipe_parameter_types_are_checked_before_processing(kind, name, value):
    graph = Graph()
    recipe = graph.recipes.put("bad", {"value": value})
    graph.recipes.activate(reference(recipe))
    block = graph.add(builtin_registry().create(kind))
    configured = copy.deepcopy(block.parameters)
    block.recipe_bindings = {name: "settings.value"}
    block.execute({"image": np.zeros((12, 16), np.uint8)})
    assert block.state == "error" and "parameter type" in block.error
    assert block.parameters == configured and block.current_emissions == {}


def test_saved_recipe_example_and_gui_version_activation(app, load_example, tmp_path):
    from PySide6.QtTest import QTest

    from opencv_dataflow.examples import recipes_graph
    from opencv_dataflow.runner import main

    registry = builtin_registry()
    graph = Graph.load("examples/flow tools/recipes/recipes.json", registry)
    assert graph.to_dict() == recipes_graph(registry).to_dict()
    for name, threshold in (("Widget A", 110), ("Widget B", 150)):
        graph.recipes.activate(name)
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values()), [
            (b.name, b.error) for b in graph.blocks.values()
        ]
        assert graph.blocks["recipe_threshold"].last_result.outputs["threshold"] == threshold
    report = tmp_path / "report.json"
    assert (
        main(
            ["examples/flow tools/recipes/recipes.json", "--repeat", "2", "--report", str(report)]
        )
        == 0
    )
    assert json.loads(report.read_text())["active_recipe"]["id"] == "Widget A"
    window = MainWindow()
    load_example(window, "recipes")
    window.root_graph.recipes.activate("Widget B", project_default=True)
    window.start_run(False)
    for _ in range(1000):
        app.processEvents()
        if window.runner is None:
            break
        QTest.qWait(10)
    assert window.runner is None
    assert window.root_graph.blocks["recipe_threshold"].last_result.outputs["threshold"] == 150
    window.show_ticket_results()
    window.ticket_results_window.search.setText("Widget B")
    assert window.ticket_results_window.table.topLevelItemCount() >= 1
    window.ticket_results_window.close()
    window.dirty = False
    window.close()
