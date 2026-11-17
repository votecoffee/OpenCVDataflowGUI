"""Original spatially image-dependent ONNX landmark fixtures with analytic source points."""

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.model_contracts import export_contract

HERE = Path(__file__).resolve().parent
HEAD_POINTS = np.array([[5.25, 6.4], [17.3, 10.2], [10.1, 4.2]])
SOURCE_POINTS = (HEAD_POINTS + 0.5) * 5 - 0.5
CASES = {
    "refined": {},
    "channels_last": {"output": "channels", "layout": "NHWC"},
    "integer": {"refinement": "none"},
    "letterbox": {"wide": True},
    "coordinates": {"output": "xy_score", "format": "xy_score"},
    "normalized_coordinates": {
        "output": "xy_normalized",
        "format": "xy_score",
        "coordinate_units": "normalized",
    },
    "subset": {"channels": [1, 0]},
    "selection": {"selection": True, "channels": [0]},
    "pose_region": {"pose": True},
    "partial": {"image": "partial", "passed": False},
    "empty": {"image": "black", "passed": False},
    "pinned": {"pinned": True},
}


def model_bytes():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
    y, x = np.mgrid[:16, :24]
    values = {
        "weights": weights,
        "axes": np.array([2, 3], np.int64),
        "xgrid": np.tile(x, (1, 3, 1, 1)).astype(np.float32),
        "ygrid": np.tile(y, (1, 3, 1, 1)).astype(np.float32),
        "epsilon": np.full((1, 3, 1, 1), 1e-10, np.float32),
        "shape": np.array([1, 3, 3], np.int64),
        "normalizer": np.tile([24, 16, 1], (1, 3, 1)).astype(np.float32),
    }
    nodes = [
        helper.make_node("Conv", ["image", "weights"], ["heatmaps"], kernel_shape=[1, 1]),
        helper.make_node("Transpose", ["heatmaps"], ["channels"], perm=[0, 2, 3, 1]),
        helper.make_node("ReduceSum", ["heatmaps", "axes"], ["sum"], keepdims=1),
        helper.make_node("Add", ["sum", "epsilon"], ["denominator"]),
        helper.make_node("Mul", ["heatmaps", "xgrid"], ["weighted_x"]),
        helper.make_node("Mul", ["heatmaps", "ygrid"], ["weighted_y"]),
        helper.make_node("ReduceSum", ["weighted_x", "axes"], ["sum_x"], keepdims=1),
        helper.make_node("ReduceSum", ["weighted_y", "axes"], ["sum_y"], keepdims=1),
        helper.make_node("Div", ["sum_x", "denominator"], ["cx"]),
        helper.make_node("Div", ["sum_y", "denominator"], ["cy"]),
        helper.make_node("ReduceMax", ["heatmaps"], ["confidence"], axes=[2, 3], keepdims=1),
        helper.make_node("Concat", ["cx", "cy", "confidence"], ["xy4"], axis=3),
        helper.make_node("Reshape", ["xy4", "shape"], ["xy_score"]),
        helper.make_node("Div", ["xy_score", "normalizer"], ["xy_normalized"]),
    ]
    graph = helper.make_graph(
        nodes,
        "original_color_landmarks",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 16, 24])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in (
                ("heatmaps", [1, 3, 16, 24]),
                ("channels", [1, 16, 24, 3]),
                ("xy_score", [1, 3, 3]),
                ("xy_normalized", [1, 3, 3]),
            )
        ],
        [numpy_helper.from_array(a, n) for n, a in values.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = (
        "Original MIT identity color heatmaps and intensity-centroid arithmetic; "
        "no learned weights."
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved; rebuild in a fresh folder")
    path.write_bytes(raw)


def build():
    asset(HERE / "assets/landmarks.onnx", model_bytes())
    images = {}
    for name in ("gaussians", "partial", "black", "wide"):
        y, x = np.mgrid[:80, : 160 if name == "wide" else 120]
        rgb = np.stack(
            [np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 7**2)) for cx, cy in SOURCE_POINTS],
            axis=-1,
        )
        if name == "partial":
            rgb[..., 2] *= 0.2
        if name == "black":
            rgb[:] = 0
        pixels = np.rint(rgb[..., ::-1] * 65535).astype(np.uint16)
        images[name] = pixels
        ok, encoded = cv2.imencode(".png", pixels)
        assert ok
        asset(HERE / f"assets/{name}.png", encoded.tobytes())
    registry, truth = builtin_registry(), {}
    for case, options in CASES.items():
        graph = Graph()
        try:
            name = options.get("image", "wide" if options.get("wide") else "gaussians")
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={
                        "path": f"assets/{name}.png",
                        "path_base": "project",
                        "pixel_depth": "unchanged",
                    },
                )
            )
            output = options.get("output", "heatmaps")
            inference = graph.add(
                registry.create(
                    "ml.dnn",
                    block_id="infer",
                    parameters={
                        "model_path": "assets/landmarks.onnx",
                        "width": 24,
                        "height": 16,
                        "scale": 1 / 65535,
                        "input_name": "image",
                        "output_names": [output],
                        "engine": "classic",
                        "resize_mode": "letterbox" if options.get("wide") else "stretch",
                    },
                )
            )
            graph.project_dir = HERE
            inference.project_dir = HERE
            if options.get("pinned"):
                with tempfile.TemporaryDirectory() as tmp:
                    bundle = Path(tmp) / "bundle"
                    export_contract(inference, bundle)
                    for filename in ("model.onnx", "contract.json"):
                        asset(HERE / "bundle" / filename, (bundle / filename).read_bytes())
                inference.parameters.update(model_path="", contract_path="bundle/contract.json")
            channels = options.get("channels", [0, 1, 2])
            params = {
                "output_name": output,
                "names": [["red", "green", "blue"][i] for i in channels],
                "channels": options.get("channels", []),
                "connections": [[i, i + 1] for i in range(len(channels) - 1)],
            }
            for key in ("layout", "format", "refinement", "coordinate_units"):
                if key in options:
                    params[key] = options[key]
            decoder = graph.add(
                registry.create("ml.decode_landmarks", block_id="decode", parameters=params)
            )
            decoder.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for a, o, b, i in (
                ("source", "image", "infer", "image"),
                ("source", "image", "decode", "image"),
                ("infer", "tensors", "decode", "tensors"),
                ("infer", "preprocessing", "decode", "preprocessing"),
                ("decode", "image", "overlay", "image"),
                ("decode", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            for i, block in enumerate((source, inference, decoder, overlay, filmstrip)):
                block.position = [410 * i, 0]
            decoder.node_size = "large"
            if options.get("selection"):
                roi = graph.add(
                    registry.create(
                        "region.shape",
                        block_id="selection",
                        parameters={"rectangle": [15, 15, 40, 40]},
                    )
                )
                roi.position = [410, 430]
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "decode", "region")
            if options.get("pose"):
                pose = graph.add(registry.create("ml.landmark_pose", block_id="pose"))
                roi = graph.add(
                    registry.create(
                        "region.pose", block_id="roi", parameters={"rectangle": [-12, -5, 24, 10]}
                    )
                )
                roi.exposed_outputs.append("result_graphics")
                for a, o, b, i in (
                    ("source", "image", "pose", "image"),
                    ("decode", "landmarks", "pose", "landmarks"),
                    ("pose", "pose", "roi", "pose"),
                    ("source", "image", "roi", "image"),
                    ("roi", "result_graphics", "overlay", "graphics_2"),
                ):
                    graph.connect(a, o, b, i)
                pose.position, roi.position = [820, 430], [1230, 430]
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.node_size, note.position = (
                "Original landmark fixtures - MIT",
                "large",
                [0, -500],
            )
            note.description = (
                "Original MIT uint16 Gaussian color images, image-dependent identity/centroid "
                "ONNX, graphs and analytic landmark references by OpenCVDataflowGUI contributors. "
                "No learned recognition or customer data. Ordered rejected landmarks remain "
                "present; incomplete Points output is suppressed. Pinned project verifies "
                "model/settings bytes. Partial/empty projects intentionally fail inspection. "
                "See README/truth.json."
            )
            graph.save(HERE / f"{case}.json")
            expected = SOURCE_POINTS[channels].copy()
            if options.get("refinement") == "none":
                expected = (np.rint(HEAD_POINTS[channels]) + 0.5) * 5 - 0.5
            if options.get("format") == "xy_score":
                rgb = cv2.resize(images[name][..., ::-1].astype(np.float32), (24, 16)) * np.float32(
                    1 / 65535
                )
                yy, xx = np.mgrid[:16, :24]
                expected = np.array(
                    [
                        [
                            (rgb[..., i] * xx).sum() / rgb[..., i].sum(),
                            (rgb[..., i] * yy).sum() / rgb[..., i].sum(),
                        ]
                        for i in channels
                    ]
                )
                expected = (expected + 0.5) * 5 - 0.5
            truth[case] = {
                **options,
                "passed": options.get("passed", True),
                "channels": channels,
                "accepted": []
                if name == "black"
                else [i for i, c in enumerate(channels) if not (name == "partial" and c == 2)],
                "source_points": expected.tolist(),
                "source_tolerance": 0.06 if options.get("wide") else 0.002,
            }
        finally:
            graph.close()
    files = [*HERE.joinpath("assets").iterdir(), *HERE.joinpath("bundle").iterdir()]
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original image landmarks and numerical controls",
                    "cases": truth,
                    "asset_sha256": {
                        p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(files)
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
