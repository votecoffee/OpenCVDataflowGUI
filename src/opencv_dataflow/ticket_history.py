"""Durable ticket summaries and runtime session history."""

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .tickets import ticket_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS runtime_session (
    session TEXT PRIMARY KEY,
    revision TEXT NOT NULL,
    project TEXT NOT NULL,
    started TEXT NOT NULL,
    finished TEXT,
    status TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS ticket_result (
    revision TEXT NOT NULL,
    identity TEXT NOT NULL,
    session TEXT NOT NULL,
    updated TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL,
    PRIMARY KEY (revision, identity)
);
CREATE INDEX IF NOT EXISTS ticket_result_recent ON ticket_result(updated DESC);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path):
    return sqlite3.connect(path, timeout=2, check_same_thread=False)


def begin(path, session, revision, project):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(path)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        db.execute(
            "INSERT INTO runtime_session VALUES (?, ?, ?, ?, NULL, 'running', '')",
            (session, revision, str(project), now()),
        )


def finish(path, session, status, message):
    with closing(connect(path)) as db, db:
        db.execute(
            "UPDATE runtime_session SET status=?, message=?, finished=? WHERE session=?",
            (status, message[-4096:], now(), session),
        )
        db.execute("""DELETE FROM runtime_session WHERE session IN (
            SELECT session FROM runtime_session WHERE finished IS NOT NULL
            ORDER BY finished DESC LIMIT -1 OFFSET 1000)""")


class TicketJournal:
    """Store an updatable result for each typed ticket across worker restarts."""

    def __init__(self, path, session, revision, *, retain=2000):
        self.db = connect(path)
        self.session, self.revision, self.retain = session, revision, retain
        self.lock = threading.Lock()

    def record(self, row):
        session, kind, value = ticket_key(row["ticket"])
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        identity = json.dumps([self.session, session, kind, value], allow_nan=False)
        detail = {
            k: row[k]
            for k in (
                "ticket",
                "parent_ticket",
                "production_excluded",
                "event_count",
                "status",
                "had_error",
                "had_failure",
                "judgments",
                "finalized_at",
                "processing_time",
                "completion",
                "final_fail_reasons",
                "fail_reasons",
            )
            if k in row
        }
        status = row["status"]
        if row["had_error"]:
            status = "error"
        elif "completion" in row and status not in (
            "conflicting_final",
            "final_pass",
            "final_fail",
        ):
            status = "pass" if row["completion"]["passed"] else "fail"
        detail["status"] = status
        detail["recipes"] = row["recipes"][-8:]
        detail["events"] = row["events"][-8:]
        detail["omitted_events"] = row["event_count"] - len(detail["events"])
        encoded = json.dumps(detail, ensure_ascii=True, allow_nan=False)
        while len(encoded) > 128 * 1024 and detail["events"]:
            detail["events"].pop(0)
            detail["omitted_events"] += 1
            encoded = json.dumps(detail, ensure_ascii=True, allow_nan=False)
        if len(encoded) > 128 * 1024:
            raise ValueError("Durable ticket summary exceeds 128 KiB")
        with self.lock, self.db:
            self.db.execute(
                """INSERT INTO ticket_result VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (revision, identity) DO UPDATE SET session=excluded.session,
                updated=excluded.updated, status=excluded.status, detail=excluded.detail""",
                (self.revision, identity, self.session, now(), status, encoded),
            )

            self.db.execute(
                """DELETE FROM ticket_result WHERE rowid IN (
                    SELECT rowid FROM ticket_result ORDER BY updated DESC, rowid DESC
                    LIMIT -1 OFFSET ?)""",
                (self.retain,),
            )

    def close(self):
        self.db.close()


def recent(path, limit=50, *, sessions=False):
    """Read committed results without creating a database or importing a graph."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=2)) as db:
        db.row_factory = sqlite3.Row
        if sessions:
            rows = db.execute(
                "SELECT * FROM runtime_session ORDER BY started DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in rows]
        rows = db.execute(
            """SELECT revision, session, updated, status, detail FROM ticket_result
            ORDER BY updated DESC, rowid DESC LIMIT ?""",
            (limit,),
        )
        return [{**dict(row), "detail": json.loads(row["detail"])} for row in rows]
