"""Portable sample projects using synthetic inputs and stable node identifiers."""

from .core import Graph


def validation_graph(registry):
    """One expected part, with pass/fail fixtures and stable reason codes."""
    graph = Graph()
    source = graph.add(
        registry.create(
            "input.file",
            block_id="validation_source",
            parameters={
                "path": "validation_assets/pass.png",
                "path_base": "project",
            },
        )
    )
    inspection = graph.add(
        registry.create(
            "inspect.contours",
            block_id="validation_inspect",
            parameters={
                "expected_count": 1,
                "min_area": 20.0,
            },
        )
    )
    final = graph.add(
        registry.create(
            "inspect.final_judgment",
            block_id="validation_final",
            parameters={
                "fail_reason": "part_count",
                "reason_message": "Expected exactly one part",
            },
        )
    )
    for index, block in enumerate((source, inspection, final)):
        block.position = [index * 340, 0]
    graph.connect(source.id, "image", inspection.id, "image")
    graph.connect(inspection.id, "overlay", final.id, "value")
    graph.validation_sets = [
        {
            "id": "validation_smoke",
            "name": "One-part regression",
            "source": source.id,
            "judgment": final.id,
            "state_mode": "isolated",
            "allow_outputs": False,
            "cases": [
                {
                    "id": "known_pass",
                    "path": "validation_assets/pass.png",
                    "expectation": "must_pass",
                },
                {
                    "id": "known_fail",
                    "path": "validation_assets/fail.png",
                    "expectation": "must_fail",
                    "preferred_fail_reason": "part_count",
                },
                {
                    "id": "likely_pass",
                    "path": "validation_assets/pass.png",
                    "expectation": "should_pass",
                },
                {
                    "id": "likely_fail",
                    "path": "validation_assets/fail.png",
                    "expectation": "should_fail",
                    "preferred_fail_reason": "part_count",
                },
                {
                    "id": "unclassified",
                    "path": "validation_assets/unknown.png",
                    "expectation": "unknown",
                },
            ],
        }
    ]
    return graph


def sequence_graph(registry):
    """Stationary warmup followed by motion, with no camera or external files."""
    graph = Graph()
    source = graph.add(registry.create("input.sequence_demo", block_id="sequence_source"))
    source.position = [0, 350]
    for index, kind in enumerate(("background", "farneback", "track_points", "image_memory")):
        block = graph.add(registry.create(f"sequence.{kind}", block_id=f"sequence_{kind}"))
        block.position = [360, index * 430]
        if kind == "background":
            block.parameters["warmup"] = 3
        graph.connect(source.id, "image", block.id, "image")
    return graph


def recipes_graph(registry):
    """Switch inspection settings and geometric/ORB/SIFT models by recipe revision."""
    import numpy as np

    from .recipes import reference

    graph = Graph()
    source = graph.add(registry.create("input.pattern_demo", block_id="recipe_images"))
    source.position = [0, 250]
    source.execute({})
    shape_reference = np.asarray(source.last_result.outputs["shape_reference"])
    texture_reference = np.asarray(source.last_result.outputs["texture_reference"])
    recipes = []
    for name, mode, algorithm, threshold in (
        ("Widget A", "edges", "orb", 110),
        ("Widget B", "corners", "sift", 150),
    ):
        shape = registry.create("pattern.match_shape", parameters={"feature_mode": mode})
        texture = registry.create(
            "pattern.match_texture",
            parameters={"algorithm": algorithm, "orb_max_features": 500, "sift_max_features": 150},
        )
        models = {
            "geometry": shape.train({"image": shape_reference}),
            "surface": texture.train({"image": texture_reference}),
        }
        recipes.append(
            graph.recipes.put(
                name,
                {
                    "threshold": threshold,
                    "shape_score": 0.65 if mode == "edges" else 0.5,
                    "shape_coverage": 0.65 if mode == "edges" else 0.3,
                    "shape_distance": 3.0 if mode == "edges" else 6.0,
                    "texture_ratio": 0.4,
                },
                models,
            )
        )
    graph.recipes.activate(reference(recipes[0]), project_default=True)
    specs = [
        ("recipe_select", "recipe.select", [300, 250], None),
        ("recipe_surface", "recipe.field", [300, 650], {"path": "models.surface"}),
        ("recipe_scalar", "recipe.field", [300, 1000], {"path": "settings.threshold"}),
        (
            "recipe_shape",
            "pattern.match_shape",
            [650, 0],
            {"rotation_min": -20.0, "rotation_max": 20.0},
        ),
        ("recipe_texture", "pattern.match_texture", [650, 400], None),
        ("recipe_threshold", "cv.threshold", [650, 800], None),
        ("recipe_join", "flow.join", [1050, 250], None),
        ("recipe_final", "inspect.final_judgment", [1350, 250], None),
    ]
    for key, kind, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.position = position
    graph.blocks["recipe_shape"].recipe_model = "models.geometry"
    graph.blocks["recipe_shape"].recipe_bindings = {
        "min_score": "settings.shape_score",
        "min_coverage": "settings.shape_coverage",
        "distance_tolerance": "settings.shape_distance",
    }
    graph.blocks["recipe_texture"].recipe_bindings = {"min_inlier_ratio": "settings.texture_ratio"}
    graph.blocks["recipe_threshold"].recipe_bindings = {"value": "settings.threshold"}
    for source_id, output, target, input_name in [
        ("recipe_images", "image", "recipe_select", "value"),
        ("recipe_select", "recipe", "recipe_surface", "recipe"),
        ("recipe_select", "recipe", "recipe_scalar", "recipe"),
        ("recipe_select", "value", "recipe_shape", "image"),
        ("recipe_select", "value", "recipe_texture", "image"),
        ("recipe_select", "value", "recipe_threshold", "image"),
        ("recipe_surface", "value", "recipe_texture", "model"),
        ("recipe_shape", "matches", "recipe_join", "branch_1"),
        ("recipe_texture", "matches", "recipe_join", "branch_2"),
        ("recipe_join", "group", "recipe_final", "value"),
    ]:
        graph.connect(source_id, output, target, input_name)
    return graph


def input_coordination_graph(registry):
    """Two synthetic cameras share a simulated PLC part counter."""
    graph = Graph()
    counter = graph.add(registry.create("input.counter", block_id="coord_counter"))
    counter.name, counter.position = "Simulated PLC part counter", [0, 160]
    for index, side in enumerate(("left", "right")):
        source = graph.add(registry.create("input.demo", block_id=f"coord_{side}"))
        source.name, source.position = f"Synthetic {side} camera", [285, index * 380]
        source.ticket_settings = {"mode": "external", "value": 1}
        graph.connect(counter.id, "value", source.id, "ticket")
    coordinator = graph.add(registry.create("flow.coordinate", block_id="coord_inputs"))
    coordinator.position = [600, 160]
    for index, side in enumerate(("left", "right"), 1):
        graph.connect(f"coord_{side}", "image", coordinator.id, f"input_{index}")
    histogram = graph.add(registry.create("inspect.histogram", block_id="coord_histogram"))
    histogram.name, histogram.position = "Primary camera histogram", [915, 0]
    final = graph.add(registry.create("inspect.final_judgment", block_id="coord_final"))
    final.position = [915, 380]
    graph.connect(coordinator.id, "value", histogram.id, "image")
    graph.connect(coordinator.id, "group", final.id, "value")
    return graph


def ticket_results_graph(registry):
    """Inspection, failure-gated diagnostics and an independent final judgment."""
    graph = Graph()
    ids = ("ticket_source", "ticket_gray", "ticket_blur", "ticket_threshold", "ticket_inspect")
    types = ("input.demo", "cv.gray", "cv.gaussian_blur", "cv.threshold", "inspect.contours")
    blocks = [graph.add(registry.create(kind, block_id=key)) for key, kind in zip(ids, types)]
    blocks[3].parameters["value"] = 100
    blocks[4].parameters["min_area"] = 1000.0
    for index, block in enumerate(blocks):
        block.position = [index * 265.0, (index % 2) * 45.0]
    for previous, following in zip(blocks, blocks[1:]):
        graph.connect(previous.id, "image", following.id, "image")
    inspect = blocks[-1]
    judgment = graph.add(
        registry.create(
            "inspect.final_judgment",
            block_id="ticket_final",
            parameters={"data": {"station": "synthetic inspection"}},
        )
    )
    judgment.position = [1325, 0]
    gate = graph.add(registry.create("cv.gray", block_id="ticket_accepted_preview"))
    gate.name, gate.position = "Only inspect passing images", [1325, 340]
    gate.input_failure_policies = {"image": "reject_failed"}
    graph.connect(inspect.id, "data", judgment.id, "value")
    graph.connect(inspect.id, "overlay", gate.id, "image")
    return graph


def starter_graph(registry):
    graph = Graph()

    def add(key, type_id, name, position, parameters=None):
        block = registry.create(type_id, block_id=key, parameters=parameters)
        block.name, block.position = name, list(position)
        return graph.add(block)

    source = add("starter_source", "input.demo", "Synthetic parts", (0, 0))
    crop = add(
        "starter_crop",
        "cv.crop",
        "Inspection ROI",
        (285, 0),
        {"x": 35, "y": 65, "width": 400, "height": 220},
    )
    gray = add("starter_gray", "cv.convert", "Grayscale", (570, 0))
    median = add("starter_median", "cv.median_blur", "Remove noise", (570, 310))
    threshold = add("starter_threshold", "cv.auto_threshold", "Segment parts", (285, 310))
    morphology = add("starter_morphology", "cv.morphology", "Clean mask", (0, 310))
    components = add(
        "starter_components",
        "inspect.components",
        "Measure two parts",
        (0, 620),
        {"min_area": 1000, "expected_count": 2},
    )
    components.preview_output = "overlay"
    histogram = add("starter_histogram", "inspect.histogram", "ROI histogram", (570, 620))
    edges = add("starter_edges", "cv.canny", "Part edges", (285, 620), {"low": 40.0, "high": 120.0})
    chain = [source, crop, gray, median, threshold, morphology, components]
    for previous, following in zip(chain, chain[1:]):
        graph.connect(previous.id, "image", following.id, "image")
    graph.connect(gray.id, "image", histogram.id, "image")
    graph.connect(median.id, "image", edges.id, "image")
    return graph


def template_graph(registry):
    graph = Graph()
    source = graph.add(registry.create("input.demo", block_id="match_source"))
    template = graph.add(
        registry.create(
            "cv.crop",
            block_id="match_template",
            parameters={"x": 250, "y": 85, "width": 165, "height": 185},
        )
    )
    template.name, template.position = "Reference patch", [0, 320]
    matcher = graph.add(
        registry.create(
            "inspect.match_template", block_id="match_result", parameters={"min_score": 0.99}
        )
    )
    matcher.position = [340, 100]
    graph.connect(source.id, "image", template.id, "image")
    graph.connect(source.id, "image", matcher.id, "image")
    graph.connect(template.id, "image", matcher.id, "template")
    return graph


def lena_graph(registry):
    """An editable real-image review project; assets are relative to its JSON file."""
    from .toolblocks import Binding, ToolBlock

    graph = Graph()

    def add(key, type_id, name, position, parameters=None):
        block = registry.create(type_id, block_id=key, parameters=parameters)
        block.name, block.position = name, list(position)
        return graph.add(block)

    source = add(
        "lena_source",
        "input.file",
        "Lena - source",
        (0, 0),
        {"path": "assets/lena.jpg", "path_base": "project"},
    )
    source.exposed_outputs = ["image"]
    inner = Graph()
    gray = inner.add(registry.create("cv.convert", block_id="lena_inner_gray"))
    blur = inner.add(registry.create("cv.gaussian_blur", block_id="lena_inner_blur"))
    blur.position = [285, 0]
    inner.connect(gray.id, "image", blur.id, "image")
    preprocess = ToolBlock(
        inner,
        {"image": Binding(gray.id, "image")},
        {"gray": Binding(gray.id, "image"), "image": Binding(blur.id, "image")},
        block_id="lena_preprocess",
    )
    preprocess.name, preprocess.position = "Grayscale + blur", [310, 0]
    graph.add(preprocess)
    edges = add("lena_edges", "cv.canny", "Canny edges", (620, 0))
    color_hist = add("lena_color_hist", "inspect.histogram", "Color histogram", (0, 340))
    gray_hist = add("lena_gray_hist", "inspect.histogram", "Gray histogram", (310, 340))
    reference = add(
        "lena_reference",
        "cv.crop",
        "Face reference patch",
        (310, 680),
        {"x": 220, "y": 200, "width": 170, "height": 190},
    )
    reference.exposed_outputs = ["image"]
    match = add(
        "lena_match", "inspect.match_template", "Locate reference", (620, 340), {"min_score": 0.99}
    )
    match.exposed_outputs = ["overlay", "score", "center"]
    graph.connect(source.id, "image", preprocess.id, "image")
    graph.connect(preprocess.id, "image", edges.id, "image")
    graph.connect(source.id, "image", color_hist.id, "image")
    graph.connect(preprocess.id, "gray", gray_hist.id, "image")
    graph.connect(source.id, "image", reference.id, "image")
    graph.connect(source.id, "image", match.id, "image")
    graph.connect(reference.id, "image", match.id, "template")
    return graph


def lena_features_graph(registry):
    graph = Graph()
    specs = [
        (
            "feature_source",
            "input.file",
            "Lena source",
            [0, 0],
            {"path": "assets/lena.jpg", "path_base": "project"},
        ),
        ("feature_gray", "cv.convert", "Grayscale", [300, 0], None),
        ("feature_clahe", "cv.clahe", "Local contrast", [600, 0], None),
        ("feature_corners", "inspect.corners", "Trackable corners", [900, 0], None),
        ("feature_histogram", "inspect.histogram", "Contrast histogram", [300, 340], None),
        ("feature_edges", "cv.canny", "Contrast edges", [600, 340], None),
        ("feature_lines", "inspect.hough_lines", "Line segments", [900, 340], None),
    ]
    for key, type_id, name, position, parameters in specs:
        block = registry.create(type_id, block_id=key, parameters=parameters)
        block.name, block.position = name, position
        graph.add(block)
    for source, target in [
        ("feature_source", "feature_gray"),
        ("feature_gray", "feature_clahe"),
        ("feature_clahe", "feature_corners"),
        ("feature_clahe", "feature_histogram"),
        ("feature_clahe", "feature_edges"),
        ("feature_edges", "feature_lines"),
    ]:
        graph.connect(source, "image", target, "image")
    graph.blocks["feature_corners"].preview_output = "overlay"
    graph.blocks["feature_lines"].preview_output = "overlay"
    return graph


def lena_registration_graph(registry):
    """Feature matching against a known rotation, with a warped-image comparison."""
    graph = Graph()
    specs = [
        (
            "registration_source",
            "input.file",
            "Lena source",
            [0, 0],
            {"path": "assets/lena.jpg", "path_base": "project"},
        ),
        (
            "registration_rotate",
            "cv.rotate",
            "Known 8 degree rotation",
            [310, 0],
            {"angle_degrees": 8.0},
        ),
        (
            "registration_query",
            "inspect.orb",
            "Original ORB features",
            [0, 340],
            {"max_features": 750},
        ),
        (
            "registration_train",
            "inspect.orb",
            "Rotated ORB features",
            [310, 340],
            {"max_features": 750},
        ),
        ("registration_match", "inspect.match_orb", "Match descriptors", [620, 340], None),
        (
            "registration_estimate",
            "inspect.homography",
            "Estimate original to rotated",
            [930, 340],
            {"min_inliers": 20, "min_inlier_ratio": 0.5},
        ),
        ("registration_warp", "cv.warp_perspective", "Warp original with estimate", [930, 0], None),
        ("registration_difference", "cv.absdiff", "Registration difference", [1240, 0], None),
    ]
    for key, type_id, name, position, parameters in specs:
        block = registry.create(type_id, block_id=key, parameters=parameters)
        block.name, block.position = name, position
        graph.add(block)
    for source, output, target, input in [
        ("registration_source", "image", "registration_rotate", "image"),
        ("registration_source", "image", "registration_query", "image"),
        ("registration_rotate", "image", "registration_train", "image"),
        ("registration_query", "features", "registration_match", "query"),
        ("registration_train", "features", "registration_match", "train"),
        ("registration_match", "matches", "registration_estimate", "matches"),
        ("registration_source", "image", "registration_warp", "image"),
        ("registration_estimate", "transform", "registration_warp", "transform"),
        ("registration_rotate", "image", "registration_difference", "image_a"),
        ("registration_warp", "image", "registration_difference", "image_b"),
    ]:
        graph.connect(source, output, target, input)
    for key in ("registration_query", "registration_train"):
        graph.blocks[key].preview_output = "overlay"
    return graph


def regions_graph(registry):
    """Regions select measurements and processing without moving image coordinates."""
    graph = Graph()
    specs = [
        ("roi_source", "input.demo", "Two synthetic parts", [0, 0], None),
        ("roi_gray", "cv.convert", "Grayscale", [300, 0], None),
        ("roi_threshold", "cv.threshold", "Segment both parts", [600, 0], {"value": 100.0}),
        (
            "roi_circle",
            "region.shape",
            "Circle selects left part",
            [0, 360],
            {"shape": "circle", "center": [140, 180], "radius": 80},
        ),
        (
            "roi_rectangle",
            "region.shape",
            "Rectangle selects right part",
            [300, 360],
            {"rectangle": [260, 95, 150, 170]},
        ),
        (
            "roi_polygon",
            "region.shape",
            "Polygon excludes text",
            [600, 360],
            {"shape": "polygon", "vertices": [[50, 60], [420, 70], [420, 280], [50, 280]]},
        ),
        ("roi_left", "inspect.components", "Measure left part", [900, 0], {"expected_count": 1}),
        ("roi_right", "inspect.contours", "Inspect right part", [900, 360], {"expected_count": 1}),
        ("roi_histogram", "inspect.histogram", "Histogram inside polygon", [1200, 360], None),
        ("roi_blur", "cv.gaussian_blur", "Blur only inside circle", [1200, 0], {"kernel": 15}),
    ]
    for key, kind, name, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
    for source, output, target, input in [
        ("roi_source", "image", "roi_gray", "image"),
        ("roi_gray", "image", "roi_threshold", "image"),
        *[
            ("roi_source", "image", key, "image")
            for key in ("roi_circle", "roi_rectangle", "roi_polygon", "roi_histogram", "roi_blur")
        ],
        ("roi_threshold", "image", "roi_left", "image"),
        ("roi_circle", "region", "roi_left", "region"),
        ("roi_threshold", "image", "roi_right", "image"),
        ("roi_rectangle", "mask", "roi_right", "mask"),
        ("roi_polygon", "region", "roi_histogram", "region"),
        ("roi_circle", "region", "roi_blur", "region"),
    ]:
        graph.connect(source, output, target, input)
    for key in ("roi_circle", "roi_rectangle", "roi_polygon", "roi_left", "roi_right"):
        graph.blocks[key].preview_output = "overlay"
    return graph


def combined_regions_graph(registry):
    """A circular selection with a hole, unioned with a second inspection region."""
    graph = Graph()
    specs = [
        ("combined_source", "input.demo", "Synthetic parts", [0, 0], None),
        (
            "combined_outer",
            "region.shape",
            "Outer circle",
            [300, 0],
            {"shape": "circle", "center": [140, 180], "radius": 80},
        ),
        (
            "combined_inner",
            "region.shape",
            "Exclude center",
            [300, 340],
            {"shape": "circle", "center": [140, 180], "radius": 28},
        ),
        (
            "combined_rectangle",
            "region.shape",
            "Second part",
            [300, 680],
            {"rectangle": [260, 95, 150, 170]},
        ),
        (
            "combined_ring",
            "region.combine",
            "Circle minus center",
            [620, 0],
            {"operation": "difference"},
        ),
        ("combined_union", "region.combine", "Ring or second part", [940, 0], None),
        (
            "combined_measure",
            "region.measure",
            "Selection size",
            [1260, 340],
            {"min_area": 40000, "max_area": 45000},
        ),
        ("combined_histogram", "inspect.histogram", "Combined histogram", [1260, 0], None),
        ("combined_gray", "cv.convert", "Grayscale", [620, 680], None),
        (
            "combined_threshold",
            "cv.threshold",
            "Segment selected pixels",
            [940, 680],
            {"value": 100.0},
        ),
        (
            "combined_components",
            "inspect.components",
            "Measure ring and rectangle",
            [1260, 680],
            {"region_mode": "clip", "expected_count": 2},
        ),
    ]
    for key, kind, name, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
    for key in (
        "combined_outer",
        "combined_inner",
        "combined_rectangle",
        "combined_ring",
        "combined_union",
        "combined_measure",
        "combined_histogram",
        "combined_gray",
    ):
        graph.connect("combined_source", "image", key, "image")
    for source, output, target, input in [
        ("combined_outer", "region", "combined_ring", "region_a"),
        ("combined_inner", "region", "combined_ring", "region_b"),
        ("combined_ring", "region", "combined_union", "region_a"),
        ("combined_rectangle", "region", "combined_union", "region_b"),
        *[
            ("combined_union", "region", key, "region")
            for key in (
                "combined_measure",
                "combined_histogram",
                "combined_threshold",
                "combined_components",
            )
        ],
        ("combined_gray", "image", "combined_threshold", "image"),
        ("combined_threshold", "image", "combined_components", "image"),
    ]:
        graph.connect(source, output, target, input)
    for key in (
        "combined_outer",
        "combined_inner",
        "combined_rectangle",
        "combined_ring",
        "combined_union",
        "combined_components",
    ):
        graph.blocks[key].preview_output = "overlay"
    return graph


def transformed_regions_graph(registry):
    """Carry one inspection region through crop, scale and rotation geometry."""
    graph = Graph()
    specs = [
        ("tracked_source", "input.demo", "Original image", [0, 0], None),
        (
            "tracked_shape",
            "region.shape",
            "Left-part region",
            [0, 350],
            {"shape": "circle", "center": [140, 180], "radius": 80},
        ),
        (
            "tracked_crop",
            "cv.crop",
            "Crop image",
            [300, 0],
            {"x": 50, "y": 60, "width": 380, "height": 240},
        ),
        ("tracked_crop_region", "region.transform", "Region in crop", [300, 350], None),
        ("tracked_resize", "cv.resize", "Resize image", [600, 0], {"width": 570, "height": 360}),
        ("tracked_resize_region", "region.transform", "Region in resized image", [600, 350], None),
        ("tracked_rotate", "cv.rotate", "Rotate image", [900, 0], {"angle_degrees": 10.0}),
        ("tracked_rotate_region", "region.transform", "Region follows rotation", [900, 350], None),
        ("tracked_histogram", "inspect.histogram", "Histogram in moved region", [1200, 0], None),
        ("tracked_threshold", "cv.threshold", "Segment moved part", [1200, 350], {"value": 100.0}),
        (
            "tracked_components",
            "inspect.components",
            "Inspect one moved part",
            [1500, 350],
            {"expected_count": 1, "region_mode": "clip"},
        ),
        (
            "tracked_measure",
            "region.measure",
            "Measure transformed selection",
            [1500, 0],
            {"min_area": 44000, "max_area": 46000},
        ),
    ]
    for key, kind, name, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
    graph.connect("tracked_source", "image", "tracked_shape", "image")
    previous_image, previous_region = "tracked_source", "tracked_shape"
    for image, region in [
        ("tracked_crop", "tracked_crop_region"),
        ("tracked_resize", "tracked_resize_region"),
        ("tracked_rotate", "tracked_rotate_region"),
    ]:
        graph.connect(previous_image, "image", image, "image")
        graph.connect(image, "image", region, "image")
        graph.connect(image, "geometry", region, "transform")
        graph.connect(previous_region, "region", region, "region")
        previous_image, previous_region = image, region
    for key in ("tracked_histogram", "tracked_threshold", "tracked_measure"):
        graph.connect(previous_image, "image", key, "image")
        graph.connect(previous_region, "region", key, "region")
    graph.connect("tracked_threshold", "image", "tracked_components", "image")
    graph.connect(previous_region, "region", "tracked_components", "region")
    for key in (
        "tracked_shape",
        "tracked_crop_region",
        "tracked_resize_region",
        "tracked_rotate_region",
        "tracked_components",
    ):
        graph.blocks[key].preview_output = "overlay"
    return graph


def channel_pyramid_graph(registry):
    """Channel round-trip, bundled levels and explicitly mapped small-level measurements."""
    graph = Graph()
    specs = [
        ("pyr_source", "input.demo", "Synthetic parts", [0, 0], None),
        (
            "pyr_crop",
            "cv.crop",
            "Inspection image",
            [300, 0],
            {"x": 50, "y": 70, "width": 380, "height": 220},
        ),
        ("pyr_split", "cv.split_channels", "Split BGR channels", [600, 0], None),
        ("pyr_merge", "cv.merge_channels", "Merge BGR channels", [900, 0], None),
        (
            "pyr_payload",
            "core.attach_payload",
            "Attach recipe payload",
            [0, 350],
            {"payload": {"recipe": {"name": "Pyramid inspection", "revision": 1}}},
        ),
        ("pyr_set", "cv.image_pyramid", "Four image levels", [300, 350], {"levels": 4}),
        ("pyr_threshold", "cv.threshold", "Threshold at level 2", [600, 350], None),
        ("pyr_components", "inspect.components", "Blobs in level-2 coordinates", [900, 350], None),
        ("pyr_coordinates", "core.read_payload", "Mapping to largest image", [1200, 350], None),
        ("pyr_centroids", "cv.transform_points", "Centroids in largest image", [1500, 350], None),
        ("pyr_preview", "cv.pyramid_level", "Preview level 1", [300, 700], {"level": "1"}),
        ("pyr_histogram", "inspect.histogram", "Largest level by default", [600, 700], None),
    ]
    for key, kind, name, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
    graph.blocks["pyr_threshold"].input_levels = {"image": "2"}
    for source, output, target, input in [
        ("pyr_source", "image", "pyr_crop", "image"),
        ("pyr_crop", "image", "pyr_split", "image"),
        *[("pyr_split", f"channel_{i}", "pyr_merge", f"channel_{i}") for i in range(3)],
        ("pyr_merge", "image", "pyr_payload", "value"),
        ("pyr_payload", "value", "pyr_set", "image"),
        ("pyr_set", "images", "pyr_threshold", "image"),
        ("pyr_threshold", "image", "pyr_components", "image"),
        ("pyr_components", "centroids", "pyr_coordinates", "value"),
        ("pyr_coordinates", "value", "pyr_centroids", "points"),
        ("pyr_coordinates", "transform", "pyr_centroids", "transform"),
        ("pyr_set", "images", "pyr_preview", "images"),
        ("pyr_set", "images", "pyr_histogram", "image"),
    ]:
        graph.connect(source, output, target, input)
    graph.blocks["pyr_components"].preview_output = "overlay"
    return graph


def object_collections_graph(registry):
    """Measured blobs, round contours and long vertical lines with visual ranked results."""
    graph = Graph()
    specs = [
        ("objects_source", "input.demo", "Synthetic parts", [0, -330], None),
        ("objects_roi", "cv.crop", "Exclude caption", [0, 0], {"width": 480, "height": 290}),
        ("objects_gray", "cv.convert", "Grayscale", [300, 0], None),
        ("objects_threshold", "cv.threshold", "Segment shapes", [600, 0], None),
        ("objects_blobs", "inspect.blobs", "Find all blobs", [0, 360], None),
        (
            "objects_blob_filter",
            "collection.filter",
            "Keep tall parts",
            [300, 360],
            {"metric": "height", "minimum": 80.0},
        ),
        (
            "objects_blob_sort",
            "collection.sort",
            "Order parts left to right",
            [600, 360],
            {"metric": "center_x", "direction": "ascending"},
        ),
        ("objects_blob_draw", "collection.draw", "Ranked parts", [900, 360], None),
        ("objects_contours", "inspect.contours", "Find contours", [0, 720], None),
        (
            "objects_contour_filter",
            "collection.filter",
            "Keep round contour",
            [300, 720],
            {"metric": "circularity", "minimum": 0.85},
        ),
        (
            "objects_contour_draw",
            "collection.draw",
            "Round contour",
            [600, 720],
            {"color": [80, 180, 250]},
        ),
        ("objects_lines", "inspect.line_segments", "Find line segments", [0, 1080], None),
        (
            "objects_line_length",
            "collection.filter",
            "Keep long lines",
            [300, 1080],
            {"metric": "length", "minimum": 100.0},
        ),
        (
            "objects_line_angle",
            "collection.filter",
            "Keep vertical lines",
            [600, 1080],
            {"metric": "angle", "minimum": 85.0, "maximum": 95.0, "use_maximum": True},
        ),
        (
            "objects_line_sort",
            "collection.sort",
            "Order lines left to right",
            [900, 1080],
            {"metric": "center_x", "direction": "ascending"},
        ),
        (
            "objects_line_draw",
            "collection.draw",
            "Vertical line results",
            [1200, 1080],
            {"color": [240, 160, 70]},
        ),
    ]
    for key, kind, name, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
    for source, output, target, input in [
        ("objects_source", "image", "objects_roi", "image"),
        ("objects_roi", "image", "objects_gray", "image"),
        ("objects_gray", "image", "objects_threshold", "image"),
        ("objects_threshold", "image", "objects_blobs", "image"),
        ("objects_blobs", "collection", "objects_blob_filter", "collection"),
        ("objects_blob_filter", "collection", "objects_blob_sort", "collection"),
        ("objects_blob_sort", "collection", "objects_blob_draw", "collection"),
        ("objects_threshold", "image", "objects_contours", "image"),
        ("objects_contours", "collection", "objects_contour_filter", "collection"),
        ("objects_contour_filter", "collection", "objects_contour_draw", "collection"),
        ("objects_gray", "image", "objects_lines", "image"),
        ("objects_lines", "collection", "objects_line_length", "collection"),
        ("objects_line_length", "collection", "objects_line_angle", "collection"),
        ("objects_line_angle", "collection", "objects_line_sort", "collection"),
        ("objects_line_sort", "collection", "objects_line_draw", "collection"),
        *[
            ("objects_roi", "image", key, "image")
            for key in ("objects_blob_draw", "objects_contour_draw", "objects_line_draw")
        ],
    ]:
        graph.connect(source, output, target, input)
    for key in ("objects_blob_draw", "objects_contour_draw", "objects_line_draw"):
        graph.blocks[key].preview_output = "overlay"
    return graph


def trained_patterns_graph(registry):
    """Separate reference training, geometric/texture search and oriented edge inspection."""
    graph = Graph()
    specs = [
        ("patterns_images", "input.pattern_demo", "References and search scene", [0, 0], None),
        (
            "patterns_shape_train",
            "pattern.train_shape",
            "Train geometric reference",
            [320, 0],
            None,
        ),
        (
            "patterns_shape_match",
            "pattern.match_shape",
            "Find rotated geometry",
            [660, 0],
            {"rotation_min": -20.0, "rotation_max": 20.0, "rotation_step": 5.0, "min_score": 0.7},
        ),
        (
            "patterns_texture_train",
            "pattern.train_texture",
            "Train ORB reference",
            [320, 400],
            None,
        ),
        (
            "patterns_texture_match",
            "pattern.match_texture",
            "Find textured reference",
            [660, 400],
            None,
        ),
        (
            "patterns_edge_region",
            "region.shape",
            "Inspect right-hand bar",
            [320, 800],
            {"rectangle": [565, 30, 60, 340]},
        ),
        (
            "patterns_edges",
            "inspect.oriented_edges",
            "Strong continuous vertical edges",
            [660, 800],
            {
                "min_length": 250.0,
                "min_continuous_length": 250.0,
                "max_thickness": 6.0,
                "min_contrast": 150.0,
            },
        ),
        (
            "patterns_edge_overlay",
            "collection.draw",
            "Measured edges",
            [1000, 800],
            {"draw_bounds": False, "color": [240, 180, 60]},
        ),
    ]
    for key, kind, name, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
    for source, output, target, input in [
        ("patterns_images", "shape_reference", "patterns_shape_train", "image"),
        ("patterns_images", "texture_reference", "patterns_texture_train", "image"),
        ("patterns_images", "image", "patterns_shape_match", "image"),
        ("patterns_images", "image", "patterns_texture_match", "image"),
        ("patterns_shape_train", "model", "patterns_shape_match", "model"),
        ("patterns_texture_train", "model", "patterns_texture_match", "model"),
        ("patterns_images", "image", "patterns_edge_region", "image"),
        ("patterns_images", "image", "patterns_edges", "image"),
        ("patterns_edge_region", "region", "patterns_edges", "region"),
        ("patterns_images", "image", "patterns_edge_overlay", "image"),
        ("patterns_edges", "collection", "patterns_edge_overlay", "collection"),
    ]:
        graph.connect(source, output, target, input)
    for key in (
        "patterns_shape_match",
        "patterns_texture_match",
        "patterns_edges",
        "patterns_edge_overlay",
    ):
        graph.blocks[key].preview_output = "overlay"
    return graph


def pose_inspection_graph(registry):
    """Locate geometry, offset to a circle, then inspect a pose-relative region."""
    graph = Graph()
    specs = [
        ("pose_images", "input.pattern_demo", "References and scene", [0, 0], None),
        ("pose_train", "pattern.train_shape", "Train geometry", [300, 0], None),
        (
            "pose_match",
            "pattern.match_shape",
            "Find geometry and circle pose",
            [620, 0],
            {
                "rotation_min": -20.0,
                "rotation_max": 20.0,
                "rotation_step": 5.0,
                "min_score": 0.7,
                "pose_offset_x": 20.5,
                "pose_offset_y": -17.5,
            },
        ),
        ("pose_offset", "pose.offset", "Adjust inspection origin", [960, 0], None),
        (
            "pose_region",
            "region.pose",
            "Circle-relative inspection",
            [620, 420],
            {"shape": "circle", "radius": 18},
        ),
        ("pose_histogram", "inspect.histogram", "Inspect positioned circle", [960, 420], None),
        (
            "pose_mask",
            "mask.shape",
            "Draw a local mask",
            [620, 820],
            {"rectangle": [-18, -18, 36, 36]},
        ),
        ("pose_mask_histogram", "inspect.histogram", "Inspect editable mask", [960, 820], None),
    ]
    for key, kind, name, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
    for source, output, target, input in [
        ("pose_images", "shape_reference", "pose_train", "image"),
        ("pose_images", "image", "pose_match", "image"),
        ("pose_train", "model", "pose_match", "model"),
        ("pose_match", "pose", "pose_offset", "pose"),
        ("pose_offset", "pose", "pose_region", "pose"),
        ("pose_images", "image", "pose_region", "image"),
        ("pose_images", "image", "pose_histogram", "image"),
        ("pose_region", "region", "pose_histogram", "region"),
        ("pose_images", "image", "pose_mask", "image"),
        ("pose_offset", "pose", "pose_mask", "pose"),
        ("pose_images", "image", "pose_mask_histogram", "image"),
        ("pose_mask", "mask", "pose_mask_histogram", "mask"),
    ]:
        graph.connect(source, output, target, input)
    graph.blocks["pose_match"].preview_output = "overlay"
    graph.blocks["pose_region"].preview_output = "overlay"
    graph.blocks["pose_mask"].preview_output = "overlay"
    return graph


def control_flow_graph(registry):
    """Three acquisitions, grouping, parallel work, routing and explicit FIFO restoration."""
    graph = Graph()
    specs = [
        ("flow_cam1", "input.demo", "Synthetic camera 1", [0, 0], {}),
        ("flow_cam2", "input.demo", "Synthetic camera 2", [0, 350], {}),
        ("flow_cam3", "input.demo", "Synthetic camera 3", [0, 700], {}),
        ("flow_batch", "flow.batch", "Group three cameras", [300, 350], {"distinct_sources": True}),
        ("flow_expected", "flow.unbatch", "Original FIFO order", [620, 0], {}),
        (
            "flow_reverse",
            "flow.unbatch",
            "Exercise reversed arrivals",
            [620, 550],
            {"order": "sequence_descending"},
        ),
        ("flow_split", "flow.split", "Two parallel branches", [940, 550], {}),
        ("flow_gray", "cv.gray", "Grayscale branch", [1260, 350], {}),
        ("flow_blur", "cv.gaussian_blur", "Blur branch", [1260, 750], {}),
        ("flow_join", "flow.join", "Synchronize same ticket", [1580, 550], {}),
        ("flow_select", "flow.select_entry", "Use grayscale result", [1900, 550], {}),
        (
            "flow_case",
            "flow.case",
            "Route even and odd tickets",
            [2220, 550],
            {"cases": [0, 1], "modulo": 2},
        ),
        ("flow_even", "cv.flip", "Even ticket branch", [2540, 350], {"direction": "horizontal"}),
        ("flow_odd", "cv.flip", "Odd ticket branch", [2540, 750], {"direction": "vertical"}),
        ("flow_fifo", "flow.reorder", "Restore acquisition FIFO", [2860, 200], {}),
        ("flow_histogram", "inspect.histogram", "Inspect ordered frames", [3180, 200], {}),
    ]
    for key, kind, name, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
    wires = [(f"flow_cam{i}", "image", "flow_batch", f"item_{i}") for i in range(1, 4)]
    wires += [
        ("flow_batch", "group", "flow_expected", "group"),
        ("flow_batch", "group", "flow_reverse", "group"),
        ("flow_reverse", "value", "flow_split", "value"),
        ("flow_split", "branch_1", "flow_gray", "image"),
        ("flow_split", "branch_2", "flow_blur", "image"),
        ("flow_gray", "image", "flow_join", "branch_1"),
        ("flow_blur", "image", "flow_join", "branch_2"),
        ("flow_join", "group", "flow_select", "group"),
        ("flow_select", "value", "flow_case", "value"),
        ("flow_case", "case_1", "flow_even", "image"),
        ("flow_case", "case_2", "flow_odd", "image"),
        ("flow_expected", "value", "flow_fifo", "expected"),
        ("flow_even", "image", "flow_fifo", "value_1"),
        ("flow_odd", "image", "flow_fifo", "value_2"),
        ("flow_fifo", "value", "flow_histogram", "image"),
    ]
    for source, output, target, input in wires:
        graph.connect(source, output, target, input)
    return graph


def parallel_parts_graph(registry):
    """Find 25 repeated patterns and inspect a local region around each matched pose."""
    graph = Graph()
    source = graph.add(registry.create("input.parts_demo", block_id="parts_source"))
    train = graph.add(registry.create("pattern.train_shape", block_id="parts_train"))
    match = graph.add(
        registry.create(
            "pattern.match_shape",
            block_id="parts_match",
            parameters={"max_matches": 25, "min_score": 0.95, "min_coverage": 0.95},
        )
    )
    inspect = graph.add(registry.create("flow.inspect_each", block_id="parts_inspect"))
    source.name, train.name, match.name, inspect.name = (
        "25 synthetic parts",
        "Train fixed geometry",
        "Locate all 25 parts",
        "Inspect every part",
    )
    source.position, train.position = [0, 0], [320, 0]
    match.position, inspect.position = [640, 0], [980, 0]
    graph.connect(source.id, "reference", train.id, "image")
    graph.connect(source.id, "image", match.id, "image")
    graph.connect(train.id, "model", match.id, "model")
    graph.connect(source.id, "image", inspect.id, "image")
    graph.connect(match.id, "matches", inspect.id, "features")
    match.preview_output = inspect.preview_output = "overlay"
    return graph


def queued_input_graph(registry):
    """Capture synthetic frames while one frame's 25 part inspections are running."""
    graph = parallel_parts_graph(registry)
    queue = graph.add(
        registry.create(
            "flow.image_queue",
            block_id="input_queue",
            parameters={
                "mode": "background_acquisition",
                "capacity": 4,
                "capture_interval_ms": 5.0,
            },
        )
    )
    queue.name, queue.position = "FIFO input queue", [320, 440]
    graph.connections = [
        c
        for c in graph.connections
        if not (c.source == "parts_source" and c.output == "image" and c.input == "image")
    ]
    graph.connect("parts_source", "image", queue.id, "image")
    graph.connect(queue.id, "image", "parts_match", "image")
    graph.connect(queue.id, "image", "parts_inspect", "image")
    return graph
