"""Saved validation image sets, isolated program execution and scored JSON reports."""

import argparse
import copy
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .blocks import ImageFile, builtin_registry
from .core import Graph, Result
from .fail_reasons import merge_reasons
from .payloads import NO_VALUE

EXPECTATIONS = ("must_pass", "should_pass", "unknown", "should_fail", "must_fail")


def validate_sets(sets):
    if not isinstance(sets, list) or len(sets) > 64:
        raise ValueError("Validation catalog must contain at most 64 sets")
    ids, total = set(), 0
    for item in sets:
        if not isinstance(item, dict) or not set(item) <= {
            "id",
            "name",
            "source",
            "judgment",
            "state_mode",
            "allow_outputs",
            "cases",
        }:
            raise ValueError("Invalid validation set")
        for key in ("id", "name", "source"):
            if not isinstance(item.get(key), str) or not 1 <= len(item[key]) <= 200:
                raise ValueError(f"Validation set requires {key}")
        if item["id"] in ids:
            raise ValueError("Duplicate validation set ID")
        ids.add(item["id"])
        if item.get("state_mode", "isolated") not in ("isolated", "sequence"):
            raise ValueError("Validation state mode must be isolated or sequence")
        if type(item.get("allow_outputs", False)) is not bool or not isinstance(
            item.get("judgment", ""), str
        ):
            raise ValueError("Invalid validation output/decision configuration")
        cases = item.get("cases")
        if not isinstance(cases, list):
            raise ValueError("Validation set requires cases")
        case_ids = set()
        for case in cases:
            if not isinstance(case, dict) or not set(case) <= {
                "id",
                "path",
                "expectation",
                "preferred_fail_reason",
                "notes",
            }:
                raise ValueError("Invalid validation image case")
            for key, maximum in (("id", 128), ("path", 4096)):
                if not isinstance(case.get(key), str) or not 1 <= len(case[key]) <= maximum:
                    raise ValueError(f"Validation image requires {key}")
            if case["id"] in case_ids or case.get("expectation") not in EXPECTATIONS:
                raise ValueError("Duplicate image ID or invalid expectation")
            case_ids.add(case["id"])
            reason = case.get("preferred_fail_reason", "")
            if not isinstance(reason, str) or len(reason) > 128:
                raise ValueError("Preferred fail reason must be a code up to 128 characters")
            if reason and case["expectation"] not in ("must_fail", "should_fail"):
                raise ValueError("Preferred fail reason applies only to expected failures")
            if not isinstance(case.get("notes", ""), str) or len(case.get("notes", "")) > 4096:
                raise ValueError("Validation notes exceed 4096 characters")
        total += len(cases)
    if total > 10000:
        raise ValueError("Validation catalog exceeds 10,000 images")
    return copy.deepcopy(sets)


def walk(graph):
    for block in graph.blocks.values():
        yield block
        if hasattr(block, "graph"):
            yield from walk(block.graph)


def prepare_graph(definition, manifest, registry, project_dir):
    graph = Graph.from_dict(copy.deepcopy(definition), registry, project_dir=project_dir)
    old = graph.blocks.get(manifest["source"])
    if old is None or not old.ticket_source or not any(p.name == "image" for p in old.outputs):
        raise ValueError("Choose a root image source for validation")
    used = {c.output for c in graph.connections if c.source == old.id}
    if not used <= {"image", "info", "ticket"}:
        raise ValueError(
            "Selected source has additional outputs; use a dedicated file/camera image source"
        )
    source = ImageFile(
        block_id=old.id,
        parameters={
            "color_format": old.parameters.get("color_format", "bgr"),
            "pixel_depth": old.parameters.get("pixel_depth", "uint8"),
        },
    )
    source.name, source.position, source.project_dir = old.name, old.position, graph.project_dir
    graph.blocks[old.id] = source
    graph.connections = [c for c in graph.connections if c.target != old.id]
    decision = manifest.get("judgment", "")
    if decision and (
        decision not in graph.blocks or graph.blocks[decision].type_id != "inspect.final_judgment"
    ):
        raise ValueError("Choose a root Final ticket judgment or automatic terminal results")
    for block in walk(graph):
        if block.type_id == "input.camera":
            raise ValueError(
                "Validation cannot open other cameras; remove or replace those sources"
            )
        if block.type_id == "flow.image_queue":
            block.parameters["mode"] = "buffer_only"
        if not manifest.get("allow_outputs", False) and (
            block.category == "Output" or block.block_type == "Output"
        ):
            block.validation_skip_output = True
            block.process = lambda inputs, b=block: Result(
                {p.name: NO_VALUE for p in b.outputs},
                message="Validation skipped output side effects",
            )
    graph.bind_recipes()
    graph.bind_activity()
    return graph


def score_report(rows, cancelled=False):
    graded = [r for r in rows if r["expectation"] != "unknown"]
    must = [r for r in rows if r["expectation"].startswith("must_")]
    advisory = [r for r in rows if r["expectation"].startswith("should_")]
    errors = sum(bool(r["errors"]) for r in rows)
    return {
        "completed": len(rows),
        "scored": len(graded),
        "unknown": len(rows) - len(graded),
        "matched": sum(r["matched"] is True for r in graded),
        "score_percent": 100 * sum(r["matched"] is True for r in graded) / len(graded)
        if graded
        else None,
        "must_mismatches": sum(r["matched"] is not True for r in must),
        "advisory_mismatches": sum(r["matched"] is not True for r in advisory),
        "execution_errors": errors,
        "cancelled": cancelled,
        "passed": bool(rows)
        and not cancelled
        and not errors
        and all(r["matched"] is True for r in must),
    }


def run_set(
    definition,
    manifest,
    registry,
    project_dir=None,
    cancelled=lambda: False,
    progress=lambda row: None,
    filmstrip=None,
):
    manifest = validate_sets([manifest])[0]
    root = Path(project_dir or Path.cwd())
    report = {
        "model": "validation_report_v1",
        "set": copy.deepcopy(manifest),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "opencv": cv2.__version__,
        "program_digest": hashlib.sha256(
            json.dumps(definition, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "cases": [],
        "cleanup_errors": [],
    }
    run_id, graph = uuid.uuid4().hex, None
    try:
        for case in manifest["cases"]:
            if cancelled():
                break
            started = time.perf_counter()
            row = {
                **copy.deepcopy(case),
                "actual": None,
                "fail_reasons": [],
                "errors": [],
                "matched": None,
                "reason_matched": None,
                "processing_ms": None,
            }
            try:
                if graph is None or manifest.get("state_mode", "isolated") == "isolated":
                    if graph is not None:
                        graph.close()
                    graph = prepare_graph(definition, manifest, registry, root)
                    if filmstrip is not None:
                        graph.filmstrip, graph.filmstrip_enabled = filmstrip, True
                path = Path(case["path"])
                path = path if path.is_absolute() else root / path
                data = path.read_bytes()
                row["image_sha256"] = hashlib.sha256(data).hexdigest()
                source = graph.blocks[manifest["source"]]
                source.parameters["path"] = str(path.resolve())
                source.read_bytes = lambda path, frozen=data: frozen
                source.ticket_settings = {
                    "mode": "fixed",
                    "value": f"validation:{run_id}:{case['id']}",
                }
                graph.run(cancelled)
                ticket = (
                    source.last_result.outputs["ticket"]
                    if source.last_result
                    else source._source_ticket
                )
                row["ticket"] = ticket
                history = graph.ticket_results.find(ticket) if ticket else None
                row["errors"] = [f"{b.name}: {b.error}" for b in walk(graph) if b.state == "error"]
                if history:
                    decisions = history["judgments"]
                    if manifest.get("judgment"):
                        decisions = list(
                            {
                                e["final_judgment"]
                                for e in history["events"]
                                if e["source"].get("path") == manifest["judgment"]
                                and e["final_judgment"] is not None
                            }
                        )
                    if len(decisions) == 1:
                        row["actual"] = decisions[0]
                    elif len(decisions) > 1:
                        row["errors"].append("Conflicting final judgments")
                    elif not manifest.get("judgment"):
                        row["actual"] = history.get("completion", {}).get("passed")
                    row["fail_reasons"] = history.get(
                        "final_fail_reasons", history.get("fail_reasons", [])
                    )
                    if manifest.get("judgment"):
                        row["fail_reasons"] = merge_reasons(
                            *(
                                e.get("fail_reasons", [])
                                for e in history["events"]
                                if e["source"].get("path") == manifest["judgment"]
                                and e["final_judgment"] is False
                            )
                        )
                    row["processing_ms"] = history.get(
                        "completion", history.get("processing_time", {})
                    ).get("elapsed_ms")
            except Exception as exc:
                row["errors"].append(f"{type(exc).__name__}: {exc}")
            if case["expectation"] != "unknown":
                expected = case["expectation"].endswith("_pass")
                preferred = case.get("preferred_fail_reason", "")
                row["reason_matched"] = (
                    preferred in {r["code"] for r in row["fail_reasons"]} if preferred else None
                )
                row["matched"] = (
                    row["actual"] is expected
                    and not row["errors"]
                    and row["reason_matched"] is not False
                )
            row["wall_ms"] = (time.perf_counter() - started) * 1000
            report["cases"].append(row)
            progress(copy.deepcopy(row))
    finally:
        if graph is not None:
            try:
                graph.close()
            except Exception as exc:
                report["cleanup_errors"].append(f"{type(exc).__name__}: {exc}")
    report["summary"] = score_report(report["cases"], cancelled())
    if report["cleanup_errors"]:
        report["summary"]["passed"] = False
        report["summary"]["execution_errors"] += len(report["cleanup_errors"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--set", required=True, dest="set_id", help="Set ID or unique name")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--plugin",
        type=Path,
        action="append",
        default=[],
        help="Load a trusted Python block plugin before opening the program",
    )
    args = parser.parse_args(argv)
    try:
        if args.report.exists():
            raise FileExistsError(f"Report already exists: {args.report}")
        registry = builtin_registry()
        from .plugins import load_plugin

        for plugin in args.plugin:
            load_plugin(plugin, registry)
        graph = Graph.load(args.project, registry)
        matches = [s for s in graph.validation_sets if args.set_id in (s["id"], s["name"])]
        if len(matches) != 1:
            raise ValueError("Select one validation set by ID or unique name")
        report = run_set(graph.to_dict(), matches[0], registry, graph.project_dir)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["summary"]))
        return 0 if report["summary"]["passed"] else 1
    except Exception as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
