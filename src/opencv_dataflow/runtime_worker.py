"""Worker for the operator runtime. Its supervisor owns deadlines and process lifetime."""

import hashlib
import json
import sys
import time
from pathlib import Path

from .diagnostics import exception_message


def execute(config, cancelled, publish):
    from .assistant_api import make_registry, validate_graph
    from .core import Graph
    from .outcomes import TicketResults
    from .queued_execution import QueuedExecution, background_queue
    from .replay import replay_sources
    from .ticket_history import TicketJournal

    project = Path(config["project"])
    raw = project.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["revision"]:
        raise ValueError("Project changed before worker startup")
    registry = make_registry(config["plugins"])
    graph = Graph.from_dict(json.loads(raw), registry, project_dir=project.parent)
    queued = None
    journal = None
    failed = False
    cycles = 0
    try:
        validation = validate_graph(graph)
        if not validation["valid"]:
            errors = [item["message"] for item in validation["issues"]]
            raise ValueError("Invalid project: " + " | ".join(errors[:8]))
        options = config["options"]
        mode = options["mode"]
        repeat = {"all": "stop", "continuous": "loop"}.get(mode)
        if repeat:
            for source in replay_sources(graph):
                source.configure_replay(repeat)
        single_cycle = mode == "once" or (mode == "all" and not graph.has_finite_sources())
        journal = TicketJournal(
            config["history"], config["session"], config["revision"], retain=options["retention"]
        )
        graph.bind_ticket_results(TicketResults(journal=journal))
        graph.filmstrip_enabled = False
        if background_queue(graph) is not None:
            queued = QueuedExecution(graph, registry, cancelled, repeat=repeat)
            queued.start()
        else:
            graph.start_program(cancelled)
        while not cancelled():
            publish("cycle", f"Cycle {cycles + 1} | acquiring or processing")
            if queued is not None:
                if not queued.run_one():
                    break
            elif graph.sources_exhausted():
                break
            else:
                graph.run(cancelled)
            if cancelled():
                break
            cycles += 1
            blocks = list(graph.processing_blocks())
            errors = [f"{b.name}: {b.error}" for b in blocks if b.state in ("error", "skipped")]
            if errors:
                raise RuntimeError(" | ".join(errors[:8]))
            stats = graph.ticket_results.statistics()
            failed |= any(b.state == "fail" for b in blocks)
            publish(
                "idle", f"Cycle {cycles} complete | ticket statistics {json.dumps(stats)}"
            )
            if single_cycle:
                break
            if mode == "continuous":
                time.sleep(options["interval_ms"] / 1000)
    finally:
        publish("shutdown", "Closing graph and acquisition")
        try:
            if queued is not None:
                queued.close()
        finally:
            try:
                graph.close()
            finally:
                if journal is not None:
                    journal.close()
    if graph.program_status()["error"]:
        raise RuntimeError(graph.program_status()["error"])
    if cancelled():
        return 130, "Stopped"
    stats = graph.ticket_results.statistics()
    return int(failed), f"Completed {cycles} cycle(s) | ticket statistics {json.dumps(stats)}"


def main():
    settings = Path(sys.argv[1])
    sys.stdout.reconfigure(encoding="utf-8")
    config = json.loads(settings.read_text(encoding="utf-8"))
    marker = settings.with_name("progress.json")
    pending = marker.with_suffix(".tmp")
    sequence = 0

    def publish(phase, message, code=None):
        nonlocal sequence
        sequence += 1
        data = dict(phase=phase, message=str(message)[:1600], code=code, sequence=sequence)
        pending.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        # Windows readers can briefly hold the target open without sharing delete.
        for attempt in range(6):
            try:
                pending.replace(marker)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.01)

    cancel = settings.with_name("stop").exists
    try:
        code, message = execute(config, cancel, publish)
    except InterruptedError as exc:
        code, message = (130, "Stopped") if cancel() else (2, exception_message(exc))
    except Exception as exc:
        code, message = 2, exception_message(exc)
    publish("done", message, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
