"""Replay projects and score observed tracks without feeding truth into the graph."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.tracking_evaluation import evaluate_tracking

HERE = Path(__file__).resolve().parent


def evaluate(case, profile="debug", *, directory=None):
    directory = HERE if directory is None else Path(directory)
    truth = json.loads((directory / "truth.json").read_bytes())
    annotation = truth["sequences"][truth["cases"][case]["sequence"]]
    frames, predictions, states = [], [], []
    with ProjectSession.open(directory / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        for frame in annotation:
            assert (
                hashlib.sha256((directory / frame["path"]).read_bytes()).hexdigest()
                == frame["sha256"]
            )
            session.graph.run()
            tool = session.graph.blocks["track"]
            if tool.state == "error" or not tool.current_emissions:
                raise RuntimeError(tool.error or "Tracker emitted no current output")
            items = tool.last_result.outputs["tracks"]["items"]
            current = []
            for item in items:
                if not item["observed"]:
                    continue
                points = np.asarray(item["geometry"]["points"])
                low, high = points.min(axis=0), points.max(axis=0)
                current.append(
                    {
                        "id": item["track_id"],
                        "label": item["label"],
                        "bbox": np.r_[low, high - low].tolist(),
                    }
                )
            frames.append({"truth": frame["truth"], "predictions": current})
            predictions.append(sum(not item["observed"] for item in items))
            states.append(
                {
                    "track_ids": [i["track_id"] for i in items],
                    "observed": [i["observed"] for i in items],
                    "source_ids": [i["source_detection_id"] for i in items],
                    "report": tool.last_result.outputs["report"],
                }
            )
    return {
        "case": case,
        "profile": profile,
        "evaluation": evaluate_tracking(frames, iou_threshold=0.6),
        "predicted_counts": predictions,
        "states": states,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=json.loads((HERE / "truth.json").read_bytes())["cases"])
    parser.add_argument("--profile", choices=("debug", "balanced", "lean"), default="debug")
    parser.add_argument("--report", type=Path, help="Fresh output JSON filename")
    args = parser.parse_args()
    report = evaluate(args.case, args.profile)
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report["evaluation"]["summary"], allow_nan=False))
