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
