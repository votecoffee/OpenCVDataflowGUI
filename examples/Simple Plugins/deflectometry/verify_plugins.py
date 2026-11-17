"""Check both explicit plugin load orders in separate fresh interpreters."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def child(first, second):
    from opencv_dataflow.assistant_api import ProjectSession, make_registry
    from opencv_dataflow.plugins import load_plugin

    registry = make_registry()
    load_plugin(first, registry)
    load_plugin(second, registry)
    with ProjectSession(registry) as session:
        session.apply(
            [
                dict(op="add", type="3d.simple_fringe_scan", id="fringe"),
                dict(op="add", type="image.simple_deflectometry", id="deflect"),
            ]
        )
        assert session.inspect()["validation"]["valid"]
        assert session.graph.blocks["deflect"].cache.model is None
    print(json.dumps(dict(both_registered=True, untrained_validation_without_processing=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fringe", type=Path)
    parser.add_argument("deflect", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.fringe.resolve(), args.deflect.resolve())
    else:
        if args.receipt is None or args.receipt.exists():
            raise ValueError("Choose a new receipt")
        cases = []
        for first, second in ((args.fringe, args.deflect), (args.deflect, args.fringe)):
            run = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(first), str(second), "--child"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            cases.append(
                dict(first=first.parent.name, second=second.parent.name, **json.loads(run.stdout))
            )
        result = dict(
            load_orders=cases,
            fringe_plugin_sha256=hashlib.sha256(args.fringe.read_bytes()).hexdigest(),
            deflect_plugin_sha256=hashlib.sha256(args.deflect.read_bytes()).hexdigest(),
            passed=True,
        )
        args.receipt.write_bytes((json.dumps(result, indent=2) + "\n").encode())
        print(json.dumps(dict(load_orders=2, passed=True)))
