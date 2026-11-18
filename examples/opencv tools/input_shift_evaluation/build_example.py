"""Paired exposure/blur/pose controls around the local anomaly inspection workflow."""

import hashlib
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "local_anomaly"
CASES = (
    "clean_stable",
    "defect_stable",
    "high_depth",
    "exposure_sensitive",
    "blur_sensitive",
    "pose_sensitive",
    "loose_threshold",
    "mixed_parts",
    "missing_variant",
    "wrong_identity",
    "cropped_truth",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def specifications(case):
    if case in ("clean_stable", "defect_stable", "high_depth"):
        return {"baseline": {}, "exposure": {"offset": 0.02}, "pose": {"translation_x": 2}}
    if case == "mixed_parts":
        return {"baseline": {}, "pose": {"translation_x": 2}}
    kind = (
        "exposure"
        if case in ("exposure_sensitive", "loose_threshold", "missing_variant", "wrong_identity")
        else "blur"
        if case == "blur_sensitive"
        else "pose"
    )
    options = (
        {"offset": 0.25}
        if case == "exposure_sensitive"
        else {"blur_kernel": 3, "blur_sigma": 1}
        if kind == "blur"
        else {"translation_x": 20 if case == "cropped_truth" else 1}
        if kind == "pose"
        else {"offset": 0.02}
    )
    return {"baseline": {}, kind: options}


def build(destination=HERE):
    destination = Path(destination)
    if (destination / "assets").exists():
        raise FileExistsError("Existing example assets preserved")
    for relative in [
        *(
            "assets/" + name
            for name in (
                "normal.png",
                "defect.png",
                "high16.png",
                "reference.json",
                "train-0.png",
                "train-1.png",
                "train-2.png",
            )
        ),
        "bundle/contract.json",
        "bundle/model.onnx",
    ]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ORIGINAL / relative).read_bytes())
    changed = json.loads((destination / "assets/reference.json").read_bytes())
    changed["samples"][0][0] = float(np.float32(changed["samples"][0][0] + 0.01))
    changed["revision"] = model_revision(changed)
    (destination / "assets/changed-reference.json").write_bytes(encoded(changed))
    registry, cases = builtin_registry(), {}
    for case in CASES:
        graph = Graph()
        graph.project_dir = destination

        def add(kind, name, x, y=0, **p):
            block = graph.add(registry.create(kind, block_id=name, parameters=p))
            block.position = [x, y]
            return block

        specs = specifications(case)
        subjects = (
            ["normal.png", "defect.png"]
            if case == "mixed_parts"
            else [
                "high16.png"
                if case == "high_depth"
                else "defect.png"
                if case in ("defect_stable", "loose_threshold", "cropped_truth")
                else "normal.png"
            ]
        )
        try:
            required = list(specs) + (["pose"] if case == "missing_variant" else [])
            total = len(subjects) * len(specs)
            batch = add("flow.batch", "batch", 2800, 500, size=total)
            metrics = add(
                "ml.evaluate_input_shifts", "metrics", 3150, 500, required_variants=required
            )
            graph.connect(batch.id, "group", metrics.id, "observations")
            count = 0
            for subject_index, image_name in enumerate(subjects):
                defective = image_name != "normal.png"
                prefix = "s" + str(subject_index) + "_"
                source = add(
                    "input.file",
                    prefix + "source",
                    0,
                    subject_index * 1600,
                    source_mode="files",
                    paths=["assets/" + image_name],
                    path_base="project",
                    color_format="unchanged",
                    pixel_depth="unchanged",
                )
                truth = add(
                    "region.shape",
                    prefix + "truth",
                    350,
                    subject_index * 1600,
                    rectangle=[12, 10, 8, 8] if defective else [40, 40, 1, 1],
                )
                graph.connect(source.id, "image", truth.id, "image")
                for index, (variant, parameters) in enumerate(specs.items()):
                    name = prefix + variant + "_"
                    y = subject_index * 1600 + index * 500
                    shift = add(
                        "ml.image_input_shift",
                        name + "shift",
                        700,
                        y,
                        variant=variant,
                        expected_pass=not defective,
                        **parameters,
                    )
                    convert = add(
                        "cv.convert",
                        name + "convert",
                        1050,
                        y,
                        color_format="gray",
                        pixel_depth="uint8",
                    )
                    net = add(
                        "ml.dnn",
                        name + "net",
                        1400,
                        y,
                        contract_path="bundle/contract.json",
                        width=32,
                        height=32,
                        color_format="gray",
                        input_name="image",
                        output_names=["local"],
                        mean=[0.0],
                        std=[1.0],
                        engine="classic",
                        receipt_mode="detailed",
                    )
                    net.exposed_outputs.append("receipt")
                    features = add(
                        "ml.local_features", name + "features", 1750, y, output_name="local"
                    )
                    reference = add(
                        "ml.load_local_anomaly",
                        name + "reference",
                        1750,
                        y + 220,
                        path="assets/"
                        + (
                            "changed-reference.json"
                            if case == "wrong_identity" and variant != "baseline"
                            else "reference.json"
                        ),
                    )
                    inspect = add(
                        "ml.local_anomaly",
                        name + "inspect",
                        2100,
                        y,
                        distance_threshold=0.95 if case == "loose_threshold" else 0.2,
                    )
                    inspect.exposed_outputs.append("result_graphics")
                    capture = add("ml.capture_shift_observation", name + "capture", 2450, y)
                    for a, o, b, i in [
                        (source, "image", shift, "image"),
                        (truth, "mask", shift, "truth"),
                        (shift, "image", convert, "image"),
                        (convert, "image", net, "image"),
                        (shift, "image", features, "image"),
                        (net, "tensors", features, "tensors"),
                        (net, "preprocessing", features, "preprocessing"),
                        (net, "receipt", features, "receipt"),
                        (shift, "image", inspect, "image"),
                        (features, "features", inspect, "features"),
                        (reference, "model", inspect, "model"),
                        (shift, "support", inspect, "mask"),
                        (shift, "image", capture, "image"),
                        (shift, "truth", capture, "truth"),
                        (shift, "support", capture, "support"),
                        (inspect, "mask", capture, "prediction"),
                        (inspect, "support", capture, "model_support"),
                        (inspect, "report", capture, "report"),
                        (inspect, "maximum_distance", capture, "score"),
                    ]:
                        graph.connect(a.id, o, b.id, i)
                    count += 1
                    graph.connect(capture.id, "observation", batch.id, f"item_{count}")
                    if subject_index == 0 and variant != "baseline":
                        overlay = add("cv.overlay_result_graphics", name + "overlay", 2800, y)
                        film = add("output.filmstrip_image", name + "filmstrip", 3150, y)
                        graph.connect(inspect.id, "image", overlay.id, "image")
                        graph.connect(inspect.id, "result_graphics", overlay.id, "graphics_1")
                        graph.connect(overlay.id, "image", film.id, "image")
            note = add("layout.comment", "attribution", 0, -650)
            note.name, note.node_size = "Original paired input-shift evaluation - MIT", "large"
            note.description = (
                "Original OpenCVDataflowGUI synthetic patch surfaces/model/reference bank, MIT. "
                "Known defect geometry supplies independent masks. Source seeds are disjoint "
                "from training. Derived exposure, blur and affine pose images are paired views "
                "of the same acquisition, not independent captures. Border support is excluded "
                "from inspection and evaluation. Scores are raw distances. "
                "See README.md/SOURCES.md."
            )
            graph.save(destination / (case + ".json"))
            target = "s0_pose_capture" if case == "cropped_truth" else "metrics"
            cases[case] = {
                "subjects": subjects,
                "variants": specs,
                "target": target,
                "required_variants": required,
                "observation_count": total,
            }
        finally:
            graph.close()
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "asset_sha256": {
                    p.relative_to(destination).as_posix(): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sorted(
                        [*(destination / "assets").iterdir(), *(destination / "bundle").iterdir()]
                    )
                },
            }
        )
    )


if __name__ == "__main__":
    build()
