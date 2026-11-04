"""Portable sample projects using synthetic inputs and stable node identifiers."""

from .core import Graph


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
    edges = add(
        "starter_edges", "gapi.canny", "Part edges", (285, 620), {"low": 40.0, "high": 120.0}
    )
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
    blur = inner.add(registry.create("gapi.blur", block_id="lena_inner_blur"))
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
    edges = add("lena_edges", "gapi.canny", "Canny edges", (620, 0))
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
        ("feature_edges", "gapi.canny", "Contrast edges", [600, 340], None),
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
        ("roi_blur", "gapi.blur", "Blur only inside circle", [1200, 0], {"kernel": 15}),
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
