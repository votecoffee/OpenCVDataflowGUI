"""Rebuild the original synthetic classification and abstention examples."""

import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.confidence_calibration import dnn_score_source

HERE = Path(__file__).resolve().parent
CLASSES = ["red", "blue"]
RUNTIME = {
    "model_path": "model.onnx",
    "color_format": "rgb",
    "width": 24,
    "height": 24,
    "input_name": "image",
    "output_names": ["scores"],
    "receipt_mode": "detailed",
    "engine": "classic",
}
GATE = {"minimum_confidence": 0.78, "minimum_margin": 0.20}


def write_model():
    import onnx
    from onnx import TensorProto, helper

    # Freeze a deliberately overconfident, untrained rule on R/B mean intensity.
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["image"], ["mean"]),
            helper.make_node("Flatten", ["mean"], ["features"], axis=1),
            helper.make_node("Gemm", ["features", "weights", "bias"], ["scores"], transB=1),
        ],
        "synthetic_two_color_parts",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 24, 24])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])],
        [
            helper.make_tensor("weights", TensorProto.FLOAT, [2, 3], [8, 0, 0, 0, 0, 8]),
            helper.make_tensor("bias", TensorProto.FLOAT, [2], [0, 0]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="OpenCVDataflowGUI",
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    (HERE / "model.onnx").write_bytes(model.SerializeToString())


def part(winner, delta, sample):
    # 12 darker and 12 lighter columns per row preserve each exact channel mean.
    # Roll each row to produce distinct patterns across image variants.
    # Model sees BGR input converted to RGB by the inference block.
    ramp = np.where(
        (np.arange(24) + np.arange(24)[:, None] * (1 + sample % 5)) % 24 < 12,
        -12 - sample % 10,
        12 + sample % 10,
    )
    blue, red = (80, 80 + delta) if winner == "red" else (80 + delta, 80)
    image = ramp[:, :, None] + np.array([blue, 88 + sample % 12, red])
    return image.astype(np.uint8)


def record(dnn, identity, winner, delta, label, sample):
    image = part(winner, delta, sample)
    dnn.execute({"image": image})
    if dnn.state != "pass":
        raise RuntimeError(dnn.error)
    outputs = dnn.last_result.outputs
    tensor = outputs["tensors"]["scores"]
    source = dnn_score_source(outputs["receipt"], "scores", tensor)
    return {
        "id": identity,
        "capture_group": identity,
        "label": label,
        "scores": tensor["values"],
    }, source


def split(dnn, prefix, conditions, offset):
    records = []
    sources = set()
    for delta, count, correct in conditions:
        for winner, other in (CLASSES, CLASSES[::-1]):
            for i in range(count):
                row, source = record(
                    dnn,
                    f"{prefix}-{len(records):03d}",
                    winner,
                    delta,
                    winner if i < correct else other,
                    offset + len(records),
                )
                sources.add(source)
                records.append(row)
    assert len(sources) == 1
    return records, sources.pop()


def add(kind, identity, params=None):
    return {"op": "add", "type": kind, "id": identity, "parameters": params or {}}


def connect(source, output, target, input_port):
    return {
        "op": "connect",
        "source": source,
        "output": output,
        "target": target,
        "input": input_port,
    }


def save(registry, filename, operations):
    with ProjectSession(registry, directory=HERE) as session:
        session.apply([*operations, {"op": "arrange"}])
        # The examples are self-contained in this directory. JSON generated for
        # a fresh copy also works when previous examples are already present.
        data = json.dumps(session.graph.to_dict(), indent=2, allow_nan=False) + "\n"
        (HERE / filename).write_text(data, encoding="utf-8", newline="\n")


def build():
    import cv2

    write_model()
    registry = make_registry()
    dnn = registry.create("ml.dnn", parameters=RUNTIME)
    dnn.project_dir = HERE
    try:
        calibration_rows, score_source = split(dnn, "cal", [(80, 12, 9), (160, 12, 11)], 0)
        evaluation_rows, test_source = split(dnn, "test", [(60, 10, 7), (145, 10, 9)], 200)
        assert score_source == test_source
        for index, delta in enumerate([0, 8, 15, 20]):
            row, source = record(dnn, f"uncertain-{index}", "red", delta, None, 350 + index)
            assert score_source == source
            evaluation_rows.append(row)
    finally:
        dnn.close()
    contract = {"classes": CLASSES, "score_source": score_source}
    fit = {**contract, "records": calibration_rows}
    evaluator = {**contract, **GATE, "records": evaluation_rows}
    fit_node = add("ml.fit_confidence_calibration", "fit", fit)
    eval_node = add("ml.evaluate_confidence_calibration", "evaluation", evaluator)
    save(
        registry,
        "reliability.json",
        [
            fit_node,
            eval_node,
            connect("fit", "calibration", "evaluation", "calibration"),
        ],
    )
    fitter = registry.create("ml.fit_confidence_calibration", parameters=fit)
    evaluator = registry.create("ml.evaluate_confidence_calibration", parameters=evaluator)
    try:
        fitter.execute({})
        if fitter.state != "pass":
            raise RuntimeError(fitter.error)
        calibration = fitter.last_result.outputs["calibration"]
        evaluator.execute({"calibration": calibration})
        if evaluator.state != "pass":
            raise RuntimeError(evaluator.error)
        report = evaluator.last_result.outputs["report"]
    finally:
        fitter.close()
        evaluator.close()
    for name, delta in [("known_red", 155), ("ambiguous", 12)]:
        image = part("red", delta, 501)
        if not cv2.imwrite(str(HERE / (name + ".png")), image):
            raise OSError("Could not write example image")
        save(
            registry,
            name + ".json",
            [
                add("input.file", "image", {"path": name + ".png", "path_base": "project"}),
                add("ml.dnn", "dnn", RUNTIME),
                {
                    "op": "update",
                    "id": "dnn",
                    "fields": {
                        "exposed_outputs": ["tensors", "preprocessing", "receipt"],
                    },
                },
                add(
                    "ml.apply_confidence_calibration",
                    "decision",
                    {
                        "classes": CLASSES,
                        "calibration": calibration,
                        "tensor_name": "scores",
                        **GATE,
                    },
                ),
                connect("image", "image", "dnn", "image"),
                connect("dnn", "tensors", "decision", "tensors"),
                connect("dnn", "receipt", "decision", "receipt"),
            ],
        )
    print(
        json.dumps(
            {
                "temperature": calibration["temperature"],
                "raw_nll": report["raw"]["nll"],
                "calibrated_nll": report["calibrated"]["nll"],
                "raw_ece": report["raw"]["ece"],
                "calibrated_ece": report["calibrated"]["ece"],
                "known_coverage": report["known_coverage"],
                "accepted_accuracy": report["accepted_accuracy"],
                "ambiguous_accepted": report["ambiguous_accepted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
