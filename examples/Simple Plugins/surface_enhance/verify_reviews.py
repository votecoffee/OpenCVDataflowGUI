"""Reopen portable file-only map projects; no estimator or hardware runs."""

import argparse
import json
from pathlib import Path

from opencv_dataflow.assistant_api import ProjectSession, make_registry


def verify(studies):
    studies = studies.resolve()
    registry = make_registry()
    cases = json.loads((studies / "summary.json").read_bytes())["cases"]
    checks = []
    for case in cases:
        folder = studies / case["case"]
        with ProjectSession.open(folder / "project.json", registry) as session:
            assert session.inspect()["validation"]["valid"]
            sources = []
            for block in session.graph.blocks.values():
                assert block.type_id in ("input.file", "layout.comment")
                if block.type_id == "input.file":
                    assert block.parameters["path_base"] == "project"
                    relative = Path(block.parameters["path"])
                    assert not relative.is_absolute()
                    image = (folder / relative).resolve()
                    assert image.is_relative_to(folder) and image.is_file()
                    sources.append(block)
            assert len(sources) == 8
            assert session.run()["passed"]
            for block in sources:
                image = block.last_result.outputs["image"]
                assert image.dtype.name == "uint8" and image.ndim == 3 and image.shape[2] == 3
            checks.append(
                dict(
                    case=case["case"],
                    valid=True,
                    reopened=True,
                    portable_paths=True,
                    rgb_views_loaded=len(sources),
                    estimation_reexecuted=False,
                    passed=True,
                )
            )
    return dict(
        scope="File-only saved map review, not surface estimation", cases=checks, passed=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    receipt = verify(args.studies)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(cases=len(receipt["cases"]), views=40, passed=True)))
