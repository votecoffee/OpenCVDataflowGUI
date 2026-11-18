"""Opt-in local-only pretrained ONNX/reference preparation and independent qualification."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPORT = "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/9bf886bf93a505ab1cf841152ede7fe18bd2351a/model/"
EXTRA = "https://raw.githubusercontent.com/opencv/opencv_extra/c9cdf09e4d4e5e73effeabce97ef52a5c2789afc/testdata/cv/shared/"
FILES = {
    "aliked-n16rot-top1k-640.onnx": (
        EXPORT,
        "b46868de0794ea0f2d878985d99dc17d3639dfab85d852994254a4cdf0cdc0ea",
    ),
    "aliked_lightglue.onnx": (
        EXPORT,
        "a1bd99d294c4c6c08ca978af3128c85f8c0e603b04a03e4391428455d9624439",
    ),
    "box.png": (EXTRA, "1094629c1e2ebbc8a9ae2c9c63e18c1dff0f5f1481146e02f3b11a3d30feb20a"),
    "box_in_scene.png": (EXTRA, "8b0225ff76244a42bd1400c0904f8b7afea7d97b7d8115495e42e98ad347bd51"),
}


def prepare(directory, models, *, download=False):
    directory, models = Path(directory), Path(models)
    paths = {}
    for name, (prefix, digest) in FILES.items():
        path = (models if name.endswith(".onnx") else directory) / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise FileExistsError("Existing differing asset preserved: " + str(path))
        else:
            if not download:
                raise FileNotFoundError(
                    "Use --download for explicit local-only preparation: " + name
                )
            with urllib.request.urlopen(prefix + name, timeout=45) as response:
                raw = response.read(64_000_001)
            if len(raw) > 64_000_000 or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Asset byte limit/hash mismatch: " + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        paths[name] = path.resolve()
    project = directory / "reference.json"
    registry = builtin_registry()
    graph = Graph()
    try:
        for index, (side, name) in enumerate(
            (("reference", "box.png"), ("query", "box_in_scene.png"))
        ):
            image = graph.add(
                registry.create(
                    "input.file",
                    block_id=side,
                    parameters={"path": str(paths[name]), "path_base": "project"},
                )
            )
            features = graph.add(
                registry.create(
                    "ml.native_aliked_features",
                    block_id=side + "_features",
                    parameters={
                        "model_path": str(paths["aliked-n16rot-top1k-640.onnx"]),
                        "model_sha256": FILES["aliked-n16rot-top1k-640.onnx"][1],
                    },
                )
            )
            image.position, features.position = [0, index * 400], [400, index * 400]
            graph.connect(side, "image", features.id, "image")
        match = graph.add(
            registry.create(
                "ml.native_lightglue",
                block_id="match",
                parameters={
                    "model_path": str(paths["aliked_lightglue.onnx"]),
                    "model_sha256": FILES["aliked_lightglue.onnx"][1],
                },
            )
        )
        match.position = [800, 400]
        match.exposed_outputs.append("result_graphics")
        graph.connect("query", "image", "match", "image")
        for side in ("reference", "query"):
            graph.connect(side + "_features", "features", "match", side + "_features")
        note = graph.add(registry.create("layout.comment", block_id="attribution"))
        note.position, note.name = [0, -400], "Local-only OpenCV ALIKED/LightGlue references"
        note.description = (
            "OpenCV-extra box/box_in_scene images and the native ONNX exports referenced by "
            "OpenCV's ALIKED/LightGlue tests. Explicit local-only downloads, pinned by SHA-256. "
            "Matcher: cvg/LightGlue (Lindenberger et al., ICCV 2023). Extractor: ALIKED "
            "(Zhao et al., IEEE TMM 2023). Export repository has no license file; these external "
            "bytes are not redistributed in this application repo. See public SOURCES.md."
        )
        if not project.exists():
            graph.save(project)
        else:
            if json.loads(project.read_bytes()) != graph.to_dict():
                raise FileExistsError(
                    "Existing modified reference project preserved: " + str(project)
                )
        graph.run()
        result = graph.blocks["match"].last_result
        if result is None or not result.passed:
            raise AssertionError("Pretrained graph failed: " + graph.blocks["match"].error)
        features = [
            graph.blocks[side + "_features"].last_result.outputs["features"]
            for side in ("reference", "query")
        ]
        point_error = descriptor_error = 0.0
        for name, observed in zip(("box.png", "box_in_scene.png"), features, strict=True):
            image = cv2.imread(str(paths[name]))
            net = cv2.dnn.readNetFromONNX(
                str(paths["aliked-n16rot-top1k-640.onnx"]), cv2.dnn.ENGINE_NEW
            )
            net.setInput(cv2.dnn.blobFromImage(image, 1 / 255, (640, 640), swapRB=True), "image")
            xy, desc, _ = net.forward(["keypoints", "descriptors", "scores"])
            xy = (xy.reshape(-1, 2) + 1) * 0.5 * np.float32(image.shape[1::-1])
            desc = desc.reshape(-1, 128)
            desc /= np.linalg.norm(desc, axis=1, keepdims=True)
            point_error = max(
                point_error, float(np.max(np.abs(xy - np.float32(observed["points"]))))
            )
            descriptor_error = max(
                descriptor_error, float(np.max(np.abs(desc - np.float32(observed["descriptors"]))))
            )
        net = cv2.dnn.readNetFromONNX(str(paths["aliked_lightglue.onnx"]), cv2.dnn.ENGINE_NEW)
        for index, observed in enumerate(features):
            xy = np.float32(observed["points"]) / np.float32(observed["image_size"]) * 2 - 1
            net.setInput(xy[None], f"kpts{index}")
            net.setInput(np.float32(observed["descriptors"])[None], f"desc{index}")
        pairs, scores = net.forward(["matches0", "mscores0"])
        pairs, scores = pairs.reshape(-1, 2), scores.ravel()
        keep = scores >= 0.1
        pairs, scores = pairs[keep], scores[keep]
        native = cv2.LightGlueMatcher.create(
            str(paths["aliked_lightglue.onnx"]),
            0.1,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        native.setPairInfo(
            *[np.float32(f["points"]) for f in features],
            *[tuple(f["image_size"]) for f in features],
        )
        matches = native.match(*[np.float32(f["descriptors"]) for f in features])
        native_pairs = np.int64([[m.queryIdx, m.trainIdx] for m in matches])
        np.testing.assert_array_equal(native_pairs, pairs)
        native_scores = np.float32([1 - m.distance for m in matches])
        score_error = float(np.max(np.abs(native_scores - scores)))
        report = result.outputs["report"]
        np.testing.assert_array_equal(
            np.int64(report["candidate_pairs"])[np.array(report["candidate_scores"]) >= 0.1], pairs
        )
        np.testing.assert_allclose(result.outputs["matches"]["scores"], scores, atol=1e-6)
        source = np.float32(features[0]["points"])[pairs[:, 0]]
        target = np.float32(features[1]["points"])[pairs[:, 1]]
        matrix, flags = cv2.findHomography(
            source, target, cv2.RANSAC, 3, maxIters=2000, confidence=0.995
        )
        pose_error = float(
            np.max(np.abs(matrix - np.array(result.outputs["pose"]["source_to_target"])))
        )
        assert (
            point_error < 1e-4
            and descriptor_error < 1e-5
            and score_error < 1e-6
            and pose_error < 1e-7
        )
        assert int(flags.sum()) == result.outputs["inliers"]
        receipt = {
            "opencv": cv2.__version__,
            "features": [len(f["points"]) for f in features],
            "matches": len(matches),
            "inliers": result.outputs["inliers"],
            "rejected_by_score": len(report["rejected_pairs"]),
            "maximum_point_error": point_error,
            "maximum_descriptor_error": descriptor_error,
            "maximum_score_error": score_error,
            "maximum_pose_error": pose_error,
            "asset_sha256": {name: info[1] for name, info in FILES.items()},
            "model_bytes_redistributed": False,
            "scope": (
                "Pinned CPU pair; direct ONNX/native parity, "
                "not independent scene ground truth or general accuracy"
            ),
        }
        (directory / "qualification.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return receipt
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--directory", type=Path, default=ROOT / ".local/native_lightglue_reference"
    )
    parser.add_argument("--models", type=Path, default=ROOT / ".local/models")
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.models, download=args.download), indent=2))
