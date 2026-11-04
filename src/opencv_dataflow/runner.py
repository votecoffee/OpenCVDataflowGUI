"""Run a saved project without Qt, including repeatable verification receipts."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .blocks import builtin_registry
from .core import Graph
from .payloads import ImagePyramid, describe_value
from .plugins import load_plugin
from .toolblocks import ToolBlock


def describe(graph):
    records = []
    for block in graph.blocks.values():
        result = block.last_result
        record = {
            "id": block.id,
            "name": block.name,
            "type": block.type_id,
            "state": block.state,
            "elapsed_ms": block.elapsed_ms,
            "score": result.score if result else None,
            "message": block.error or (result.message if result else ""),
            "outputs": {
                p.name: describe_value(result.outputs[p.name])
                for p in block.outputs
                if p.kind not in ("image", "depth", "pyramid")
                and not isinstance(result.outputs[p.name], (np.ndarray, ImagePyramid))
            }
            if result
            else {},
            "payloads": result.payloads if result else {},
            "emitted": {name: len(packets) for name, packets in result.emissions.items()}
            if result
            else {},
        }
        if isinstance(block, ToolBlock):
            record["children"] = describe(block.graph)
        records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--repeat", type=int, default=1, help="Run the graph this many times")
    parser.add_argument(
        "--plugin",
        type=Path,
        action="append",
        default=[],
        help="Explicitly execute a trusted Python plugin before loading the project",
    )
    parser.add_argument(
        "--report", type=Path, help="Create a JSON receipt; existing files are preserved"
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    graph = None
    try:
        if args.report and args.report.exists():
            raise FileExistsError(f"Report already exists: {args.report}")
        registry = builtin_registry()
        for plugin in args.plugin:
            load_plugin(plugin, registry)
        graph = Graph.load(args.project, registry)
        runs = []
        passed = True
        for index in range(args.repeat):
            started = time.perf_counter()
            graph.run()
            records = describe(graph)
            ok = all(b.state in ("pass", "waiting", "inactive") for b in graph.blocks.values())
            passed = passed and ok
            runs.append(
                {
                    "run": index + 1,
                    "passed": ok,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "blocks": records,
                }
            )
            print(f"Run {index + 1}: {'PASS' if ok else 'FAIL'} ({len(records)} blocks)")
            for block in graph.blocks.values():
                if block.state not in ("pass", "waiting", "inactive"):
                    message = block.error or (
                        block.last_result.message if block.last_result else ""
                    )
                    print(f"  {block.name}: {block.state} · {message}")
        if args.report:
            # Ports of kind number may contain NumPy scalar objects.
            content = (
                json.dumps(
                    {"project": str(args.project), "passed": passed, "runs": runs},
                    indent=2,
                    allow_nan=False,
                    default=lambda value: value.item(),
                )
                + "\n"
            )
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(content)
        return 0 if passed else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    sys.exit(main())
