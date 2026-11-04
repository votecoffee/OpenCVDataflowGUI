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
