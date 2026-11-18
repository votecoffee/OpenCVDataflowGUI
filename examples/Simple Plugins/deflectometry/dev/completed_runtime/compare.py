"""Separate-process same-recipe parity and bounded warm-execution timing experiment."""

import argparse
import ast
import hashlib
import json
import re
import runpy
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("wheel", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.wheel.resolve()))
shared = args.root / "releases/opencv_simple_fringe_scan-0.1.0.dev3-py3-none-any.whl"
assert (
    hashlib.sha256(shared.read_bytes()).hexdigest()
    == "0106d63746e36244358454432e9f707628989134e5af2288de3e436c18737f87"
)
sys.path.insert(0, str(shared.resolve()))

from opencv_dataflow.assistant_api import ProjectSession  # noqa: E402
from opencv_dataflow.blocks import builtin_registry  # noqa: E402
from opencv_dataflow.core import Block, Port, Result  # noqa: E402
from opencv_dataflow.parameter_validation import ValidationError  # noqa: E402
from opencv_dataflow.payloads import frame  # noqa: E402
from opencv_dataflow.run_profiles import PROFILES  # noqa: E402

from opencv_simple_deflectometry.dataflow import (  # noqa: E402
    SimpleDeflectometry,
    recipe_path,
)
from opencv_simple_deflectometry.execution import capture_bundle  # noqa: E402
from opencv_simple_deflectometry.recipe import load_recipe  # noqa: E402

# Use only the recorded source's audited declaration, bypassing the current-wheel
# loader so an earlier wheel can process exactly the same saved projects.
tree = ast.parse((args.root / "plugin.py").read_text(encoding="utf-8"))
definition = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RecordedDeflectCapture"
)
namespace = dict(
    Block=Block,
    Port=Port,
    Result=Result,
    ValidationError=ValidationError,
    frame=frame,
    recipe_path=recipe_path,
    capture_bundle=capture_bundle,
    load_recipe=load_recipe,
    re=re,
)
exec(compile(ast.Module(body=[definition], type_ignores=[]), "recorded_source", "exec"), namespace)
RecordedDeflectCapture = namespace["RecordedDeflectCapture"]
registry = builtin_registry()
for typ in (SimpleDeflectometry, RecordedDeflectCapture):
    registry.types.pop(typ.type_id, None)
    registry.register(typ)
helpers = runpy.run_path(str(args.root / "examples.py"))
expected = json.loads((args.root / "expected.json").read_bytes())
originals = {
    r["case"]: r for r in json.loads((args.root / "verification.json").read_bytes())["cases"]
}
rows = []
for case in helpers["CASES"]:
    with ProjectSession.open(args.root / "projects" / f"{case}.json", registry) as session:
        session.run()
        block = session.graph.blocks["deflect"]
        assert block.last_result is not None, block.error
        assert helpers["stable_result"](block.last_result) == expected[case], case
        assert helpers["array_hashes"](block.last_result) == originals[case]["arrays"], case
        row = dict(case=case, numerical_maps_and_rasters_exact=True)
        if case in (
            "analytic_clean",
            "analytic_change",
            "connected_change",
            "missing",
            "sofast_self_review",
        ):
            block.parameters["render_graphics"] = False
            inputs, timings = block.last_inputs.copy(), {}
            for profile in ("lean", "debug"):
                block.run_profile = PROFILES[profile]
                for _ in range(2):
                    assert block.execute(inputs) is not None, block.error
                samples = []
                for _ in range(7):
                    start = time.perf_counter()
                    assert block.execute(inputs) is not None, block.error
                    samples.append(1000 * (time.perf_counter() - start))
                timings[profile] = dict(median_ms=statistics.median(samples), samples_ms=samples)
            row["warm_graphics_off_execution"] = timings
            record = getattr(block, "_completed_review", None)
            if record is not None:
                from opencv_simple_deflectometry.completed_review import retain_completed

                samples = []
                for _ in range(31):
                    start = time.perf_counter()
                    retain_completed(
                        block,
                        record._recipe,
                        record._capture,
                        record._output,
                        block.last_result.outputs,
                        record._manifest,
                    )
                    samples.append(1000 * (time.perf_counter() - start))
                row["debug_record_only_median_ms"] = statistics.median(samples)
        rows.append(row)
args.output.write_bytes(
    json.dumps(
        dict(
            wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
            cases=rows,
            scope="Same inputs/models/settings; warm graphics-off execution with host retention. "
            "Separate-process timings are exploratory, not speedup or latency guarantees.",
        ),
        indent=2,
        allow_nan=False,
    ).encode()
    + b"\n"
)
print(json.dumps(dict(cases=len(rows), all_numerical_maps_and_rasters_exact=True)))
