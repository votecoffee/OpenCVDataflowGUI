"""Supervise a saved inspection in an owned process and query durable results."""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .diagnostics import exception_message
from .managed_process import ManagedProcess
from .ticket_history import begin, finish, recent


@dataclass
class RunOptions:
    mode: str = "once"
    startup_seconds: int = 120
    cycle_seconds: int = 60
    shutdown_seconds: int = 10
    interval_ms: int = 100
    memory_mib: int = 4096
    retention: int = 2000

    def validate(self):
        if self.mode not in ("once", "all", "continuous"):
            raise ValueError("Run mode must be once, all, or continuous")
        for name, minimum, maximum in (
            ("startup_seconds", 1, 3600),
            ("cycle_seconds", 1, 7200),
            ("shutdown_seconds", 1, 120),
            ("interval_ms", 10, 60000),
            ("memory_mib", 256, 65536),
            ("retention", 1, 10000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be a whole number in {minimum}..{maximum}")


def read_progress(path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(8193)
        return json.loads(raw) if len(raw) <= 8192 else None
    except OSError, ValueError:
        return None


def run(project, history, *, options=None, plugins=(), cancelled=lambda: False, progress=print):
    """Keep native work outside the operator process and cap time per activation."""
    options = options or RunOptions()
    options.validate()
    project, history = Path(project).resolve(strict=True), Path(history).resolve()
    if history == project:
        raise ValueError("Choose a history database separate from the project")
    plugin_paths = [str(Path(plugin).resolve(strict=True)) for plugin in plugins]
    revision = hashlib.sha256(project.read_bytes()).hexdigest()
    session = uuid.uuid4().hex
    begin(history, session, revision, project)
    process = None
    status, code, message = "error", 2, "Runtime failed to start"
    with TemporaryDirectory(prefix="opencv-dataflow-operator-") as directory:
        root = Path(directory)
        settings = root / "settings.json"
        marker = root / "progress.json"
        settings.write_text(
            json.dumps(
                dict(
                    project=str(project),
                    revision=revision,
                    history=str(history),
                    session=session,
                    plugins=plugin_paths,
                    options=asdict(options),
                )
            ),
            encoding="utf-8",
        )
        try:
            # Preserve the application's environment and explicitly selected plugins.
            # Disable per-user Python site packages inside the worker.
            env = {**os.environ, "PYTHONNOUSERSITE": "1"}
            process = ManagedProcess(
                [sys.executable, "-u", "-m", "opencv_dataflow.runtime_worker", str(settings)],
                Path.cwd(),
                env,
                memory_limit=options.memory_mib * 1024 * 1024,
            )
            last = None
            phase = "startup"
            deadline = time.monotonic() + options.startup_seconds
            while True:
                update = read_progress(marker)
                if update is not None and update != last:
                    last = update
                    phase = update["phase"]
                    seconds = (
                        options.shutdown_seconds
                        if phase in ("shutdown", "done")
                        else options.cycle_seconds
                        + (options.interval_ms / 1000 if phase == "idle" else 0)
                    )
                    deadline = time.monotonic() + seconds
                    if phase != "done":
                        progress(update["message"])
                result = process.process.poll()
                if result is not None:
                    final_update = read_progress(marker)
                    if final_update is None or final_update["phase"] != "done":
                        raise RuntimeError(
                            f"Worker exited with code {result}. {process.tail()[-1500:]}"
                        )
                    if result != final_update["code"]:
                        raise RuntimeError(
                            f"Worker exited with code {result} after reporting completion"
                        )
                    code = result
                    status = {0: "pass", 1: "fail", 2: "error", 130: "stopped"}[code]
                    message = final_update["message"]
                    break
                if cancelled():
                    status, code, message = "stopped", 130, "Stop requested"
                    break
                if time.monotonic() >= deadline:
                    status, message = "timeout", f"Runtime {phase} deadline exceeded"
                    break
                time.sleep(0.025)
        except KeyboardInterrupt:
            status, code, message = "stopped", 130, "Stop requested"
        except Exception as exc:
            message = exception_message(exc)
        finally:
            try:
                if process:
                    grace = options.shutdown_seconds if status == "stopped" else 0.1
                    (root / "stop").touch()
                    process.close(timeout=grace)
            finally:
                finish(history, session, status, message)
        progress(f"{status.upper()} Ã‚Â· {message}")
    return code


def default_history():
    root = os.environ.get("LOCALAPPDATA")
    state = Path(root) if root else Path.home() / ".local" / "state"
    return state / "OpenCVDataflowGUI" / "runtime.sqlite3"


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run", help="Run a trusted saved project")
    execute.add_argument("project", type=Path)
    execute.add_argument("--mode", choices=("once", "all", "continuous"), default="once")
    execute.add_argument("--history", type=Path, default=default_history())
    execute.add_argument("--plugin", type=Path, action="append", default=[])
    defaults = asdict(RunOptions())
    for name in defaults.keys() - {"mode"}:
        execute.add_argument("--" + name.replace("_", "-"), type=int, default=defaults[name])
    query = sub.add_parser("history", help="Read committed ticket results")
    query.add_argument("--history", type=Path, default=default_history())
    query.add_argument("--limit", type=int, choices=range(1, 1001), default=50, metavar="1..1000")
    query.add_argument("--sessions", action="store_true")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "history":
            print(json.dumps(recent(args.history, args.limit, sessions=args.sessions), indent=2))
            return 0
        options = RunOptions(**{key: getattr(args, key) for key in asdict(RunOptions())})
        return run(args.project, args.history, options=options, plugins=args.plugin)
    except Exception as exc:
        print(exception_message(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
