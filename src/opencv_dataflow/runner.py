"""Run a saved project without Qt, including repeatable verification receipts."""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from .blocks import builtin_registry
from .core import Graph
from .diagnostics import exception_message
from .parallel_inspection import ParallelInspect
from .payloads import ImagePyramid, describe_value
from .plugins import load_plugin
from .queued_execution import QueuedExecution, background_queue
from .run_profiles import PROFILES
from .scheduling import MAX_BRANCH_WORKERS
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
            "recipe": block.last_recipe_ref,
            "recipe_parameters": block.last_recipe_parameters,
            "elapsed_ms": block.elapsed_ms,
            "timing": block.timing.snapshot(),
            "score": result.score if result else None,
            "message": block.error or (result.message if result else ""),
            "outputs": {
                p.name: describe_value(result.outputs[p.name])
                for p in block.execution_outputs
                if p.kind not in ("image", "depth", "pyramid")
                and not isinstance(result.outputs[p.name], (np.ndarray, ImagePyramid))
            }
            if result
            else {},
            "payloads": result.payloads if result else {},
            "emitted": {name: len(packets) for name, packets in block.current_emissions.items()},
        }
        if isinstance(block, ToolBlock):
            record["children"] = describe(block.graph)
        elif isinstance(block, ParallelInspect):
            record["inspection_preview"] = describe(block.graph)
            record["preview_index"] = block.last_preview_index
        records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--repeat", type=int, default=1, help="Run the graph this many times")
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Run a finite image or video file sequence to completion",
    )
    parser.add_argument("--profile", choices=PROFILES, help="Override the saved run profile")
    parser.add_argument(
        "--branch-workers",
        choices=["auto", *(str(n) for n in range(1, MAX_BRANCH_WORKERS + 1))],
        help="Override the root graph's saved branch worker setting",
    )
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
    queued = None
    try:
        if args.report and args.report.exists():
            raise FileExistsError(f"Report already exists: {args.report}")
        registry = builtin_registry()
        for plugin in args.plugin:
            load_plugin(plugin, registry)
        graph = Graph.load(args.project, registry)
        if args.profile:
            graph.set_run_profile(args.profile)
        if args.branch_workers is not None:
            graph.set_branch_workers(
                "auto" if args.branch_workers == "auto" else int(args.branch_workers)
            )
        if args.all_images and not graph.has_finite_sources():
            raise ValueError("--all-images requires a finite image or video source (repeat=stop)")
        if background_queue(graph) is not None:
            queued = QueuedExecution(graph, registry)
            queued.start()
        runs = []
        passed = True
        for index in itertools.count() if args.all_images else range(args.repeat):
            if queued is None and graph.sources_exhausted():
                break
            started = time.perf_counter()
            if queued is not None:
                if not queued.run_one():
                    break
            else:
                graph.run()
            records = describe(graph) if args.report else []
            ok = all(b.state in ("pass", "waiting", "inactive") for b in graph.blocks.values())
            passed = passed and ok
            if args.report:
                runs.append(
                    {
                        "run": index + 1,
                        "passed": ok,
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "blocks": records,
                    }
                )
            print(f"Run {index + 1}: {'PASS' if ok else 'FAIL'} ({len(graph.blocks)} blocks)")
            for block in graph.blocks.values():
                if block.state not in ("pass", "waiting", "inactive"):
                    message = block.error or (
                        block.last_result.message if block.last_result else ""
                    )
                    print(f"  {block.name}: {block.state} · {message}")
        if queued is not None:
            queued.close()
            queued = None
        if args.report:
            # Ports of kind number may contain NumPy scalar objects.
            content = (
                json.dumps(
                    {
                        "project": str(args.project),
                        "run_profile": graph.run_profile.name,
                        "branch_workers": graph.branch_workers,
                        "passed": passed,
                        "runs": runs,
                        "ticket_results": graph.ticket_results.snapshot(),
                        "activity": graph.activity.snapshot(),
                        "active_recipe": graph.recipes.active_ref(),
                    },
                    indent=2,
                    allow_nan=False,
                    default=lambda value: value.item(),
                )
                + "\n"
            )
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(content)
        return 0 if passed else 1
    except (Exception, SystemExit) as exc:
        print(exception_message(exc), file=sys.stderr)
        return 2
    finally:
        if queued is not None:
            queued.close()
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    sys.exit(main())
