import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED
from opencv_dataflow.run_profiles import PROFILES


def geometry_template():
    image = np.zeros((96, 112), np.uint8)
    cv2.line(image, (18, 16), (18, 70), 255, 3)
    cv2.line(image, (18, 70), (60, 70), 255, 3)
    cv2.circle(image, (76, 30), 13, 255, 2)
    cv2.rectangle(image, (47, 13), (57, 23), 255, 2)
    return image


def texture_template():
    image = np.random.default_rng(14).integers(0, 256, (192, 192), np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0.5)


def tool(kind, **parameters):
    return builtin_registry().create(kind, parameters=parameters)


def test_shape_training_translation_multiple_instances_and_no_match():
    template = geometry_template()
    matcher = tool("pattern.match_shape", max_matches=2)
    model = matcher.train({"image": template})
    assert model["model"] == "shape_pattern_v1" and len(model["points"]) >= 8
    search = np.zeros((300, 400), np.uint8)
    search[20:116, 20:132] = template
    search[160:256, 240:352] = template
    matcher.execute({"image": search})
    assert matcher.state == "pass", matcher.error
    assert matcher.last_result.outputs["count"] == 2
    np.testing.assert_allclose(
        sorted(matcher.last_result.outputs["points"]), [[75.5, 67.5], [295.5, 207.5]], atol=1
    )
    matcher.execute({"image": np.zeros_like(search)})
    assert matcher.state == "fail" and matcher.last_result.outputs["count"] == 0
    assert (
        matcher.last_result.outputs["score"] == 0 and matcher.last_result.outputs["transform"] == {}
    )


def test_shape_review_shows_below_threshold_candidates_without_emitting_them():
    template = geometry_template()
    matcher = tool("pattern.match_shape", max_matches=2)
    matcher.train({"image": template})
    search = np.zeros((220, 330), np.uint8)
    search[50:146, 25:137] = template
    search[50:146, 185:297] = template
    search[50:105, 235:297] = 0
    result = matcher.execute({"image": search})
    assert result is not None, matcher.error
    assert result.outputs["count"] == 1
    graphic = result.graphics["pattern matches"][40:]
    assert np.any(np.all(graphic[:, :155] == ACCEPTED, axis=-1))
    assert np.any(np.all(graphic[:, 155:] == REJECTED, axis=-1))
    assert not np.any(np.all(result.outputs["overlay"] == REJECTED, axis=-1))
    matcher.run_profile = PROFILES["lean"]
    lean = matcher.execute({"image": search})
    assert lean.outputs["matches"] == result.outputs["matches"]
    assert not lean.graphics


def test_shape_rotation_scale_pose_and_search_region():
    template = geometry_template()
    matcher = tool(
        "pattern.match_shape",
        rotation_min=-15.0,
        rotation_max=15.0,
        rotation_step=5.0,
        scale_min=0.9,
        scale_max=1.1,
        scale_step=0.1,
        min_score=0.65,
        distance_tolerance=4.0,
    )
    matcher.train({"image": template})
    matrix = cv2.getRotationMatrix2D((55.5, 47.5), 10, 1.1)
    matrix[:, 2] += [140, 80]
    image = cv2.warpAffine(template, matrix, (400, 300))
    matcher.execute({"image": image})
    assert matcher.state == "pass", matcher.error
    match = matcher.last_result.outputs["matches"]["items"][0]
    expected = cv2.transform(np.array([[[55.5, 47.5]]]), matrix)[0, 0]
    np.testing.assert_allclose(match["center"], expected, atol=2)
    assert abs(match["angle_degrees"] - 10) <= 5 and match["scale"] == pytest.approx(1.1)
    mask = np.zeros_like(image)
    mask[:40, :40] = 255
    matcher.execute({"image": image, "mask": mask})
    assert matcher.state == "fail" and matcher.last_result.outputs["count"] == 0


@pytest.mark.parametrize("algorithm", ["orb", "sift"])
def test_texture_training_localizes_planar_target_and_rejects_blank(algorithm):
    template = texture_template()
    matcher = tool("pattern.match_texture", algorithm=algorithm)
    model = matcher.train({"image": template})
    image = np.zeros((320, 420), np.uint8)
    image[75:267, 125:317] = template
    matcher.execute({"image": image})
    assert matcher.state == "pass", matcher.error
    assert matcher.last_result.outputs["count"] == 1
    np.testing.assert_allclose(matcher.last_result.outputs["points"][0], [220.5, 170.5], atol=2)
    assert matcher.last_result.outputs["matches"]["items"][0]["inliers"] >= 6
    assert "texture correspondences" in matcher.last_graphics
    matcher.parameters["min_inliers"] = 10000
    rejected = matcher.execute({"image": image})
    assert rejected.outputs["count"] == 0
    assert np.any(np.all(rejected.graphics["pattern matches"][50:] == REJECTED, axis=-1))
    assert not np.any(np.all(rejected.outputs["overlay"] == REJECTED, axis=-1))
    matcher.execute({"image": np.zeros_like(image)})
    assert matcher.state == "fail" and matcher.last_result.outputs["count"] == 0
    assert matcher.trained_model["id"] == model["id"]


@pytest.mark.parametrize(
    "kind,train_kind,image",
    [
        ("pattern.match_shape", "pattern.train_shape", geometry_template),
        ("pattern.match_texture", "pattern.train_texture", texture_template),
    ],
)
def test_explicit_training_blocks_and_atomic_failed_retraining(kind, train_kind, image):
    template = image()
    trainer = tool(train_kind)
    trainer.execute({"image": template})
    assert trainer.state == "pass", trainer.error
    matcher = tool(kind)
    matcher.execute({"image": template, "model": trainer.last_result.outputs["model"]})
    assert matcher.state == "pass", matcher.error
    matcher.train({"image": template})
    old = copy.deepcopy(matcher.trained_model)
    with pytest.raises(ValueError, match="features"):
        matcher.train({"image": np.zeros_like(template)})
    assert matcher.trained_model == old
    untrained = tool(kind)
    untrained.execute({"image": template})
    assert untrained.state == "error" and "Train" in untrained.error
    assert "image" in untrained.last_inputs


def test_trained_model_project_restore_and_checksum_validation(tmp_path):
    from opencv_dataflow.toolblocks import encapsulate

    template = geometry_template()
    path = tmp_path / "template.png"
    cv2.imwrite(str(path), template)
    registry, graph = builtin_registry(), Graph()
    source = graph.add(registry.create("input.file", parameters={"path": str(path)}))
    matcher = graph.add(registry.create("pattern.match_shape"))
    matcher.train({"image": template})
    graph.connect(source.id, "image", matcher.id, "image")
    nested = encapsulate(graph, [matcher.id], "Trained pattern", registry)
    project = tmp_path / "pattern.json"
    graph.save(project)
    restored = Graph.load(project, registry)
    restored.run()
    assert restored.blocks[nested.id].state == "pass", restored.blocks[nested.id].error
    loaded = restored.blocks[nested.id].graph.blocks[matcher.id]
    assert loaded.trained_model["id"] == matcher.trained_model["id"]
    bad = graph.to_dict()
    inner = next(b for b in bad["blocks"] if b["id"] == nested.id)["toolblock"]["graph"]
    inner["blocks"][0]["trained_model"]["points"][0][0] += 1
    with pytest.raises(ValueError, match="checksum"):
        Graph.from_dict(bad, registry)


@pytest.mark.parametrize("mode", ["edges", "corners", "lines", "circles", "combined"])
def test_geometric_feature_training_modes_match_their_reference(mode):
    image = geometry_template()
    matcher = tool("pattern.match_shape", feature_mode=mode, min_points=4, min_score=0.7)
    matcher.train({"image": image})
    matcher.execute({"image": image})
    assert matcher.state == "pass", matcher.error
    assert matcher.last_result.outputs["count"] == 1


def test_pattern_pose_maps_model_points_into_target_coordinate_payload():
    from opencv_dataflow.images import ImageArray
    from opencv_dataflow.patterns import model_payload
    from opencv_dataflow.payloads import ValuePacket

    template = geometry_template()
    matcher = tool("pattern.match_shape")
    model = matcher.train({"image": template})
    image = ImageArray(
        np.zeros((220, 320), np.uint8),
        "gray",
        {
            "spatial": {
                "root_id": "search-image",
                "root_size": [320, 220],
                "image_size": [320, 220],
                "to_root": np.eye(3).tolist(),
            }
        },
    )
    image[50:146, 100:212] = template
    matcher.execute({"image": image})
    assert matcher.state == "pass"
    mapper = tool("cv.transform_points")
    mapper.execute(
        {
            "points": ValuePacket([[0.0, 0.0]], model_payload(model)),
            "transform": matcher.last_result.outputs["transform"],
        }
    )
    assert mapper.state == "pass", mapper.error
    np.testing.assert_allclose(mapper.last_result.outputs["points"], [[100, 50]], atol=1)
    assert mapper.last_result.payloads["points"]["spatial"]["root_id"] == "search-image"


@pytest.mark.parametrize(
    "kind,image", [("shape", geometry_template), ("texture", texture_template)]
)
def test_training_selection_is_independent_and_models_survive_runtime_clone(kind, image):
    reference = image()
    h, w = reference.shape
    scene = np.zeros((h + 60, w + 80), np.uint8)
    scene[20 : 20 + h, 30 : 30 + w] = reference
    training_mask = np.zeros_like(scene)
    training_mask[20 : 20 + h, 30 : 30 + w] = 255
    excluded = np.zeros_like(scene)
    matcher = tool(f"pattern.match_{kind}")
    model = matcher.train({"image": scene, "mask": excluded, "training_mask": training_mask})
    assert model["training_origin"] == [30, 20]
    assert model["size"] == [w, h]
    graph = Graph()
    graph.add(matcher)
    restored = Graph.from_dict(graph.to_dict(), builtin_registry()).blocks[matcher.id]
    restored.execute({"image": scene})
    assert restored.state == "pass", restored.error
    np.testing.assert_allclose(
        restored.last_result.outputs["points"][0], [30 + (w - 1) / 2, 20 + (h - 1) / 2], atol=2
    )
    restored.execute({"image": scene, "mask": excluded})
    assert restored.state == "fail" and restored.last_result.outputs["count"] == 0


def test_saved_trained_patterns_example(tmp_path):
    from pathlib import Path

    from opencv_dataflow.examples import trained_patterns_graph

    registry = builtin_registry()
    graph = Graph.load(
        Path(__file__).parents[1]
        / "examples/custom tools/trained_patterns/trained_patterns.json",
        registry,
    )
    assert graph.to_dict() == trained_patterns_graph(registry).to_dict()
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values())
        assert graph.blocks["patterns_edges"].last_result.outputs["count"] == 2
        for name, center in (("shape", [125.5, 127.5]), ("texture", [425.5, 215.5])):
            outputs = graph.blocks[f"patterns_{name}_match"].last_result.outputs
            assert outputs["count"] == 1
            np.testing.assert_allclose(outputs["points"][0], center, atol=2)


@pytest.mark.parametrize(
    "parameters", [{"rotation_step": 0}, {"scale_min": 0}, {"min_score": 1.1}, {"max_matches": 257}]
)
def test_invalid_shape_search_parameters_clear_results(parameters):
    matcher = tool("pattern.match_shape")
    matcher.train({"image": geometry_template()})
    matcher.execute({"image": geometry_template()})
    assert matcher.state == "pass"
    matcher.parameters.update(parameters)
    matcher.execute({"image": geometry_template()})
    assert matcher.state == "error" and matcher.last_result is None
