"""JSON command interface for assistants authoring and reviewing saved projects."""

import argparse
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

import cv2

from .assistant_api import (
    API_VERSION,
    ProjectSession,
    ProjectValidationError,
    RevisionConflict,
    block_schema,
    catalog,
    make_registry,
    validate_graph,
)
from .diagnostics import exception_message
from .port_types import connection_mode


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field {key!r}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON {value}")),
    )


def example_index(root, query=""):
    """Index public runnable project files without importing code or reading private data."""
    root = Path(root).resolve()
    folders = (
        "applications",
        "demos",
        "industrial integration",
        "Simple Plugins",
        "opencv tools",
        "custom tools",
        "control flow tools",
        "skimage tools",
        "user_tools",
    )
    terms = query.casefold().split()
    records = []
    visible = None
    if (root / ".git").exists():
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "examples",
            ],
            cwd=root,
            capture_output=True,
            check=True,
        )
        visible = {
            os.fsdecode(path).replace("\\", "/") for path in result.stdout.split(b"\0") if path
        }
    for category in folders:
        for path in sorted((root / "examples" / category).rglob("*.json")):
            relative = path.relative_to(root)
            if visible is not None and relative.as_posix() not in visible:
                continue
            if any(part.casefold() in {"private", "cognextools"} for part in relative.parts):
                continue
            if not all(term in str(relative).casefold() for term in terms):
                continue
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            try:
                data = _read(path)
            except ValueError, OSError:
                continue
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                continue
            if not isinstance(data.get("blocks"), list) or "connections" not in data:
                continue
            readme = path.parent / "README.md"
            records.append(
                {
                    "project": str(path),
                    "category": category,
                    "readme": str(readme) if readme.is_file() else None,
                    "block_types": sorted({b["type"] for b in data["blocks"]}),
                }
            )
    return records


def _parser():
    parser = Parser(description=__doc__)
    parser.add_argument(
        "--plugin",
        type=Path,
        action="append",
        default=[],
        help="Explicitly execute a trusted Python plugin",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Protocol version and environment")
    p = sub.add_parser("catalog", help="Search registered and optional block IDs")
    p.add_argument("--query", default="")
    p = sub.add_parser("describe", help="Resolve one block's ports and parameter constraints")
    p.add_argument("type_id")
    p = sub.add_parser("compatibility", help="Check declared port kinds without running a block")
    p.add_argument("source_kind")
    p.add_argument("target_kind")
    p = sub.add_parser("examples", help="Find public project examples in a source checkout")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--query", default="")
    for name in ("inspect", "validate", "apply", "run", "export"):
        p = sub.add_parser(name)
        p.add_argument("project", type=Path)
        if name == "apply":
            p.add_argument("--operations", type=Path, required=True)
            p.add_argument("--output", type=Path)
            p.add_argument("--expected-revision")
            p.add_argument("--dry-run", action="store_true")
        if name == "run":
            p.add_argument("--repeat", type=int, default=1)
            p.add_argument("--all-images", action="store_true")
            p.add_argument("--max-runs", type=int, default=1000)
            p.add_argument("--profile", choices=("debug", "balanced", "lean"))
            p.add_argument("--result-graphics", choices=("inherit", "on", "off"))
            p.add_argument("--branch-workers", type=str)
            p.add_argument("--report", type=Path)
            p.add_argument("--summary", action="store_true", help="Return a compact run response")
            p.add_argument("--artifacts", type=Path, help="New directory for final review previews")
        if name == "export":
            p.add_argument("--format", choices=("html", "xml"), required=True)
            p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("new", help="Build a new project from an operation list")
    p.add_argument("--operations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--directory", type=Path, help="Original base for project-relative inputs")
    return parser


def _dispatch(args):
    if args.command == "info":
        checkout = Path(__file__).resolve().parents[2]
        checkout = checkout if (checkout / "docs" / "architecture.md").is_file() else None
        return {
            "api_version": API_VERSION,
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "application": version("opencv-dataflow-gui"),
            "transport": "local CLI / Python",
            "gui_attachment": False,
            "source_checkout": str(checkout) if checkout else None,
            "documentation": str(checkout / "docs" / "README.md") if checkout else None,
            "commands": [
                "info",
                "catalog",
                "describe",
                "examples",
                "new",
                "inspect",
                "validate",
                "apply",
                "run",
                "export",
                "compatibility",
            ],
        }, 0
    if args.command == "compatibility":
        mode = connection_mode(args.source_kind, args.target_kind)
        return {"compatible": mode is not None, "conversion": mode}, 0
    if args.command == "examples":
        return {"examples": example_index(args.root, args.query)}, 0
    registry = make_registry(args.plugin)
    if args.command == "catalog":
        return {"blocks": catalog(registry, args.query)}, 0
    if args.command == "describe":
        return block_schema(args.type_id, registry), 0
    session = (
        ProjectSession(registry, directory=args.directory or args.output.resolve().parent)
        if args.command == "new"
        else ProjectSession.open(args.project, registry)
    )
    with session:
        if args.command == "inspect":
            return session.inspect(), 0
        if args.command == "validate":
            data = validate_graph(session.graph)
            return data, 0 if data["valid"] else 1
        if args.command in ("new", "apply"):
            destination = args.output or session.source_path
            if (
                args.command == "apply"
                and destination.exists()
                and not args.dry_run
                and args.expected_revision is None
            ):
                raise RevisionConflict("Existing destination requires --expected-revision")
            if (
                args.command == "apply"
                and args.expected_revision is not None
                and destination.resolve() == session.source_path
                and session.revision != args.expected_revision
            ):
                raise RevisionConflict("Source revision changed; inspect it again")
            data = session.apply(_read(args.operations))
            if args.command == "apply" and args.dry_run:
                return {**data, "saved": False}, 0
            saved = session.save(
                destination,
                expected_revision=(args.expected_revision if args.command == "apply" else None),
            )
            return {**saved, "validation": data["validation"], "saved": True}, 0
        if args.command == "run":
            if args.report and args.report.exists():
                raise FileExistsError(f"Report already exists: {args.report}")
            if args.artifacts and args.artifacts.exists():
                raise FileExistsError(f"Artifacts directory already exists: {args.artifacts}")
            if args.profile:
                session.graph.set_run_profile(args.profile)
            if args.result_graphics:
                session.graph.set_graphics_mode(args.result_graphics)
            if args.branch_workers:
                session.graph.set_branch_workers(
                    "auto" if args.branch_workers == "auto" else int(args.branch_workers)
                )
            data = session.run(
                repeat=args.repeat, all_images=args.all_images, max_runs=args.max_runs
            )
            if args.artifacts:
                from .assistant_artifacts import export_previews

                data["artifacts"] = export_previews(session.graph, args.artifacts)
            if args.report:
                with args.report.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        json.dumps(data, indent=2, allow_nan=False, default=_json_default) + "\n"
                    )
            if args.summary:
                failures = {}

                def collect(records, scope=()):
                    for record in records:
                        if record["state"] not in ("pass", "waiting", "inactive"):
                            failures[(*scope, record["id"])] = {
                                "scope": list(scope),
                                "id": record["id"],
                                "name": record["name"],
                                "state": record["state"],
                                "message": record["message"],
                            }
                        for field in ("children", "inspection_preview"):
                            collect(record.get(field, ()), (*scope, record["id"]))

                for run in data["runs"]:
                    collect(run["blocks"])
                data = {
                    "passed": data["passed"],
                    "complete": data["complete"],
                    "stop_reason": data["stop_reason"],
                    "run_count": len(data["runs"]),
                    "failures": list(failures.values()),
                    "report": str(args.report.resolve()) if args.report else None,
                    "artifacts_manifest": (
                        str(args.artifacts.resolve() / "manifest.json") if args.artifacts else None
                    ),
                }
            return data, 0 if data["passed"] else 1
        if args.command == "export":
            if args.output.exists():
                raise FileExistsError(f"Export already exists: {args.output}")
            if args.format == "html":
                from .program_documentation import render_program_documentation

                content = render_program_documentation(session.graph, args.project.stem)
                with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
            else:
                from .project_xml import export_project_xml

                # Reserve the destination before calling the existing atomic XML exporter.
                with args.output.open("x"):
                    pass
                try:
                    export_project_xml(session.graph.to_dict(), args.output)
                except BaseException:
                    args.output.unlink(missing_ok=True)
                    raise
            return {"path": str(args.output.resolve()), "format": args.format}, 0
    raise ValueError("Unsupported command")


def main(argv=None):
    """Emit one versioned JSON response. Plugin stdout goes to stderr."""
    try:
        with redirect_stdout(sys.stderr):
            args = _parser().parse_args(argv)
            data, code = _dispatch(args)
        response = {"api_version": API_VERSION, "ok": code == 0, "data": data}
        text = json.dumps(response, allow_nan=False, default=_json_default)
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:  # --help
            return 0
        code = 3 if isinstance(exc, RevisionConflict) else 2
        text = json.dumps(
            {
                "api_version": API_VERSION,
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": exception_message(exc),
                    **({"issues": exc.issues} if isinstance(exc, ProjectValidationError) else {}),
                },
            },
            allow_nan=False,
        )
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
