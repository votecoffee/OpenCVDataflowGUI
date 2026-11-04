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
