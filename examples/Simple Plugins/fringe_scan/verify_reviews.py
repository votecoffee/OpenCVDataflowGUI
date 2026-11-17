"""Reopen portable native map/cloud reviews without rerunning reconstruction."""

import argparse
import json
from pathlib import Path

from opencv_dataflow.assistant_api import make_registry

from build_review_projects import verify_project

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    studies = args.studies.resolve()
    registry = make_registry()
    checks = [
        verify_project(studies / row["case"], row, registry)
        for row in json.loads((studies / "summary.json").read_bytes())
    ]
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(dict(cases=len(checks), native_metric_clouds=2, passed=True)))
