"""SQLite result history with opt-in text retention."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Columns kept in the ``runs`` table only so history files written earlier still
#: open. They held a former ±0.10 display band; new rows store NULL and nothing reads
#: them, and they are never returned by `get`, `list` or the CSV export.
LEGACY_COLUMNS = ("interval_low", "interval_high")

_RUNS_TABLE = """
    CREATE TABLE IF NOT EXISTS {name} (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        verdict TEXT NOT NULL,
        p_ai REAL NOT NULL,
        interval_low REAL,
        interval_high REAL,
        bundle_identity TEXT NOT NULL,
        agents_json TEXT NOT NULL,
        warnings_json TEXT NOT NULL,
        timings_json TEXT NOT NULL,
        response_json TEXT NOT NULL,
        submitted_text TEXT,
        mode TEXT NOT NULL,
        domain_hint TEXT,
        status TEXT NOT NULL DEFAULT 'succeeded',
        error_json TEXT,
        runtime_identity_json TEXT NOT NULL DEFAULT '{{}}'
    );
"""


class HistoryStore:
    """Open one short-lived connection per operation for safe API-thread use."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        redacted_existing_rows = False
        with self._connect() as connection:
            connection.executescript(
                _RUNS_TABLE.format(name="runs")
                + """
                CREATE TABLE IF NOT EXISTS feedback (
                    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                    expected_label TEXT NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            migrations = {
                "status": "ALTER TABLE runs ADD COLUMN status TEXT NOT NULL DEFAULT 'succeeded'",
                "error_json": "ALTER TABLE runs ADD COLUMN error_json TEXT",
                "runtime_identity_json": (
                    "ALTER TABLE runs ADD COLUMN runtime_identity_json TEXT NOT NULL DEFAULT '{}'"
                ),
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)
            self._relax_legacy_columns(connection)
            # Earlier versions saved the complete response even when the user declined
            # text retention. The token heatmap contains the submitted words verbatim.
            # Clean those existing rows as well as preventing future writes below.
            rows = connection.execute(
                "SELECT run_id, response_json FROM runs "
                "WHERE submitted_text IS NULL AND response_json IS NOT NULL"
            ).fetchall()
            for row in rows:
                saved = json.loads(row["response_json"])
                safe = self._response_for_storage(saved, store_text=False)
                if safe is not saved:
                    connection.execute(
                        "UPDATE runs SET response_json = ? WHERE run_id = ?",
                        (json.dumps(safe, ensure_ascii=False), row["run_id"]),
                    )
                    redacted_existing_rows = True
        if redacted_existing_rows:
            # Rewriting a SQLite value alone can leave its old bytes in free pages.
            with self._connect() as connection:
                connection.execute("VACUUM")

    @staticmethod
    def _relax_legacy_columns(connection: sqlite3.Connection) -> None:
        """Drop NOT NULL from the legacy interval columns of a earlier history file.

        SQLite cannot alter a column constraint in place, so the table is rebuilt with the
        current definition and every row copied across unchanged (old interval values
        included -- they are kept, just never read). Runs once; a file already in the new
        shape is left alone.
        """
        info = connection.execute("PRAGMA table_info(runs)").fetchall()
        if not any(str(row[1]) in LEGACY_COLUMNS and int(row[3]) for row in info):
            return
        names = ", ".join(str(row[1]) for row in info)
        connection.commit()
        connection.executescript(
            "BEGIN;"
            + _RUNS_TABLE.format(name="runs_relaxed")
            + f"INSERT INTO runs_relaxed ({names}) SELECT {names} FROM runs;"
            + "DROP TABLE runs;"
            + "ALTER TABLE runs_relaxed RENAME TO runs;"
            + "COMMIT;"
        )

    @staticmethod
    def _response_for_storage(response: dict[str, Any], *, store_text: bool) -> dict[str, Any]:
        """Keep the live response intact while removing reconstructible text from history."""
        if store_text:
            return response
        explanations = response.get("explanations")
        if not isinstance(explanations, dict) or not explanations.get("token_heatmap"):
            return response
        safe = dict(response)
        safe["explanations"] = {**explanations, "token_heatmap": []}
        return safe

    def begin(
        self,
        run_id: str,
        *,
        submitted_text: str,
        store_text: bool,
        mode: str,
        domain_hint: str | None,
        runtime_identity: dict[str, Any],
    ) -> None:
        """Persist the request before model work begins.

        Legacy columns are non-null, so honest placeholder values occupy them only
        while ``status='started'``. Consumers must use ``status`` before reading a
        verdict; successful completion atomically replaces every placeholder.
        """
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO runs (
                    run_id, created_at, verdict, p_ai,
                    bundle_identity, agents_json, warnings_json, timings_json,
                    response_json, submitted_text, mode, domain_hint, status,
                    error_json, runtime_identity_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    datetime.now(UTC).isoformat(),
                    "INCONCLUSIVE",
                    0.5,
                    json.dumps(runtime_identity, sort_keys=True, ensure_ascii=False),
                    "[]",
                    "[]",
                    "{}",
                    "{}",
                    submitted_text if store_text else None,
                    mode,
                    domain_hint,
                    "started",
                    None,
                    json.dumps(runtime_identity, sort_keys=True, ensure_ascii=False),
                ),
            )

    def finish(
        self,
        response: dict[str, Any],
        *,
        status: str = "succeeded",
        error: dict[str, Any] | None = None,
    ) -> None:
        """Finish an existing run without changing its text-retention choice."""
        if status not in {"succeeded", "refused", "failed"}:
            raise ValueError(f"invalid terminal history status: {status}")
        agents = response.get("agents", [])
        identity = response.get("runtime_identity", {})
        timings = {
            "total_ms": response.get("elapsed_ms", 0),
            "agents_ms": {row.get("name"): row.get("latency_ms") for row in agents},
        }
        with self._connect() as connection:
            row = connection.execute(
                "SELECT submitted_text FROM runs WHERE run_id = ?", (response["run_id"],)
            ).fetchone()
            if row is None:
                raise KeyError(f"history run was not started: {response['run_id']}")
            stored_response = self._response_for_storage(
                response, store_text=row["submitted_text"] is not None
            )
            cursor = connection.execute(
                """UPDATE runs SET
                    verdict = ?, p_ai = ?, interval_low = NULL, interval_high = NULL,
                    bundle_identity = ?, agents_json = ?, warnings_json = ?,
                    timings_json = ?, response_json = ?, status = ?, error_json = ?,
                    runtime_identity_json = ?
                WHERE run_id = ?""",
                (
                    response.get("verdict", "INCONCLUSIVE"),
                    response.get("p_ai", 0.5),
                    json.dumps(identity, sort_keys=True, ensure_ascii=False),
                    json.dumps(agents, ensure_ascii=False),
                    json.dumps(response.get("warnings", []), ensure_ascii=False),
                    json.dumps(timings, ensure_ascii=False),
                    json.dumps(stored_response, ensure_ascii=False),
                    status,
                    json.dumps(error, ensure_ascii=False) if error is not None else None,
                    json.dumps(identity, sort_keys=True, ensure_ascii=False),
                    response["run_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"history run was not started: {response['run_id']}")

    def save(
        self,
        response: dict[str, Any],
        *,
        submitted_text: str,
        store_text: bool,
        mode: str,
        domain_hint: str | None,
    ) -> None:
        identity = response.get("runtime_identity", response.get("model_versions", {}))
        self.begin(
            str(response["run_id"]),
            submitted_text=submitted_text,
            store_text=store_text,
            mode=mode,
            domain_hint=domain_hint,
            runtime_identity=dict(identity),
        )
        completed = dict(response)
        completed.setdefault("runtime_identity", identity)
        self.finish(completed)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT runs.*, feedback.expected_label AS feedback_expected_label,
                          feedback.comment AS feedback_comment,
                          feedback.created_at AS feedback_created_at
                   FROM runs LEFT JOIN feedback USING (run_id)
                   WHERE runs.run_id = ?""",
                (run_id,),
            ).fetchone()
        return self._row(row, include_text=True) if row is not None else None

    def list(
        self,
        *,
        verdict: str | None = None,
        domain_hint: str | None = None,
        status: str | None = None,
        mode: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[object] = []
        if verdict:
            clauses.append("verdict = ?")
            values.append(verdict)
        if domain_hint:
            clauses.append("domain_hint = ?")
            values.append(domain_hint)
        if status:
            clauses.append("status = ?")
            values.append(status)
        if mode:
            clauses.append("mode = ?")
            values.append(mode)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(max(1, min(limit, 1000)))
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT runs.*, feedback.expected_label AS feedback_expected_label,
                          feedback.comment AS feedback_comment,
                          feedback.created_at AS feedback_created_at
                   FROM runs LEFT JOIN feedback USING (run_id)"""
                + where
                + " ORDER BY runs.created_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._row(row) for row in rows]

    def add_feedback(self, run_id: str, expected_label: str, comment: str | None) -> bool:
        if self.get(run_id) is None:
            return False
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)",
                (run_id, expected_label, comment, datetime.now(UTC).isoformat()),
            )
        return True

    def export_csv(self, **filters: str | int | None) -> str:
        verdict_filter = filters.get("verdict")
        domain_filter = filters.get("domain_hint")
        status_filter = filters.get("status")
        mode_filter = filters.get("mode")
        rows = self.list(
            verdict=verdict_filter if isinstance(verdict_filter, str) else None,
            domain_hint=domain_filter if isinstance(domain_filter, str) else None,
            status=status_filter if isinstance(status_filter, str) else None,
            mode=mode_filter if isinstance(mode_filter, str) else None,
            limit=int(filters.get("limit") or 1000),
        )
        sink = io.StringIO()
        writer = csv.DictWriter(
            sink,
            fieldnames=[
                "run_id",
                "created_at",
                "verdict",
                "p_ai",
                "bundle_identity",
                "status",
                "error",
                "mode",
                "domain_hint",
                "text_stored",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], ensure_ascii=False)
                        if key == "error" and row[key] is not None
                        else row[key]
                    )
                    for key in writer.fieldnames
                }
            )
        return sink.getvalue()

    @staticmethod
    def _row(row: sqlite3.Row, *, include_text: bool = False) -> dict[str, Any]:
        result = dict(row)
        for legacy in LEGACY_COLUMNS:
            result.pop(legacy, None)
        submitted_text = result.get("submitted_text")
        result["text_stored"] = submitted_text is not None
        if submitted_text is None or not include_text:
            result.pop("submitted_text", None)
        result["agents"] = json.loads(result.pop("agents_json"))
        result["warnings"] = json.loads(result.pop("warnings_json"))
        result["timings"] = json.loads(result.pop("timings_json"))
        result["response"] = json.loads(result.pop("response_json"))
        result["runtime_identity"] = json.loads(result.pop("runtime_identity_json"))
        error = result.pop("error_json")
        result["error"] = json.loads(error) if error else None
        expected = result.pop("feedback_expected_label", None)
        comment = result.pop("feedback_comment", None)
        created = result.pop("feedback_created_at", None)
        result["feedback"] = (
            {"expected_label": expected, "comment": comment, "created_at": created}
            if expected is not None
            else None
        )
        return result
