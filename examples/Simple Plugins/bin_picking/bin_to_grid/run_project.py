"""Execute one app episode against an already running, isolated simulation fixture."""

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path

import cv2
from opencv_simple_bin_picking.sim_dataflow import register

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).with_name("project.json"))
    parser.add_argument(
        "--output", type=Path, required=True, help="New per-episode folder; never overwritten"
    )
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--execute-simulation", action="store_true", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"opencv_bin_pick_[A-Za-z0-9_-]{1,80}", args.partition):
        parser.error("Select a unique allocated simulation partition")
    profile = args.runtime_profile.resolve(strict=True)
    registry = builtin_registry()
    register(registry)
    session = ProjectSession.open(args.project, registry)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        session.apply([dict(op="update", id="pick", fields={"parameters": {"enabled": True}})])
        session.save(args.output / "project.json")
        if not session.inspect()["validation"]["valid"]:
            raise ValueError("App graph is invalid")
        os.environ.update(
            OPENCV_DATAFLOW_ROS2_PROFILE=str(profile),
            ROS_DOMAIN_ID="130",
            GZ_PARTITION=args.partition,
        )
        finished = threading.Event()

        def progress():
            last = None
            while not finished.wait(0.5):
                messages = [
                    e["message"]
                    for e in session.graph.blocks["pick"].logger.snapshot()["events"]
                    if e.get("message", "").startswith("Cell ")
                ]
                if messages and messages[-1] != last:
                    last = messages[-1]
                    print(last, file=sys.stderr, flush=True)

        thread = threading.Thread(target=progress, daemon=True)
        thread.start()
        try:
            session.run()
        finally:
            finished.set()
            thread.join(timeout=1)
        result = session.graph.blocks["summary"].last_result
        if result is None:
            raise RuntimeError("Graph did not produce a complete cell summary")
        report = result.outputs["report"]
        (args.output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        cv2.imwrite(str(args.output / "summary.png"), result.outputs["image"])
        print(
            json.dumps(
                dict(
                    output=str(args.output.resolve()),
                    complete=report["complete"],
                    placed=report["placed"],
                    cells=report["attempted_cells"],
                )
            ),
            flush=True,
        )
        return 0 if report["complete"] else 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
