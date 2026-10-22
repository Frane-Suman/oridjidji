from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from oridjidji.storage import HistoryStore

pytestmark = pytest.mark.unit


def response(run_id: str = "r1") -> dict[str, object]:
    return {
        "run_id": run_id,
        "verdict": "INCONCLUSIVE",
        "p_ai": 0.5,
        "agents": [{"name": "stub", "latency_ms": 3}],
        "warnings": ["w"],
        "model_versions": {"stub": "v1"},
        "elapsed_ms": 5,
    }


def test_history_keeps_metadata_but_text_requires_opt_in(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.save(
        response(),
        submitted_text="privatni tekst",
        store_text=False,
        mode="osnovni",
        domain_hint="news",
    )
    run = store.get("r1")
    assert run is not None
    assert run["text_stored"] is False
    assert "privatni tekst" not in (tmp_path / "history.sqlite3").read_bytes().decode(
        "utf-8", errors="ignore"
    )
    assert store.list(verdict="INCONCLUSIVE", domain_hint="news")[0]["run_id"] == "r1"


def test_feedback_retrieval_and_csv(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.save(
        response(), submitted_text="tekst", store_text=True, mode="osnovni", domain_hint=None
    )
    assert store.add_feedback("r1", "human", "provjereno")
    assert not store.add_feedback("missing", "human", None)
    saved = store.get("r1")
    assert saved is not None
    assert saved["submitted_text"] == "tekst"
    assert saved["feedback"]["expected_label"] == "human"
    exported = store.export_csv()
    assert "run_id,created_at,verdict" in exported
    assert "r1" in exported


def test_history_records_started_refused_and_failed_states(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")
    identity = {"encoder": "weights"}
    store.begin(
        "refused",
        submitted_text="tekst",
        store_text=False,
        mode="osnovni",
        domain_hint="academic",
        runtime_identity=identity,
    )
    assert store.get("refused")["status"] == "started"  # type: ignore[index]
    store.finish(
        {"run_id": "refused", "runtime_identity": identity, "elapsed_ms": 2},
        status="refused",
        error={"status_code": 422, "detail": "prekratko"},
    )
    row = store.get("refused")
    assert row is not None
    assert row["status"] == "refused"
    assert row["error"]["status_code"] == 422
    assert row["runtime_identity"] == identity
    assert "submitted_text" not in row
    assert [item["run_id"] for item in store.list(status="refused")] == ["refused"]


@pytest.mark.parametrize("use_save", [False, True])
def test_advanced_heatmap_is_not_retained_without_text_opt_in(
    tmp_path: Path, use_save: bool
) -> None:
    path = tmp_path / "history.sqlite3"
    store = HistoryStore(path)
    private_text = "Jedinstveni privatni ulaz"
    result = {
        **response(),
        "explanations": {
            "token_heatmap": [{"text": private_text, "start": 0, "end": len(private_text)}],
            "top_features": [{"name": "length", "value": 12}],
        },
    }
    if use_save:
        store.save(
            result,
            submitted_text=private_text,
            store_text=False,
            mode="napredni",
            domain_hint=None,
        )
    else:
        store.begin(
            "r1",
            submitted_text=private_text,
            store_text=False,
            mode="napredni",
            domain_hint=None,
            runtime_identity={},
        )
        store.finish(result)

    assert result["explanations"]["token_heatmap"][0]["text"] == private_text
    saved = store.get("r1")
    assert saved is not None
    assert saved["response"]["explanations"]["token_heatmap"] == []
    assert saved["response"]["explanations"]["top_features"] == [{"name": "length", "value": 12}]
    assert store.list()[0]["response"]["explanations"]["token_heatmap"] == []
    assert private_text.encode() not in path.read_bytes()


def test_advanced_heatmap_is_retained_with_explicit_text_opt_in(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")
    result = {
        **response(),
        "explanations": {"token_heatmap": [{"text": "dopušteni tekst"}]},
    }
    store.save(
        result,
        submitted_text="dopušteni tekst",
        store_text=True,
        mode="napredni",
        domain_hint=None,
    )
    saved = store.get("r1")
    assert saved is not None
    assert saved["submitted_text"] == "dopušteni tekst"
    assert saved["response"]["explanations"]["token_heatmap"] == [{"text": "dopušteni tekst"}]


def test_existing_non_opt_in_heatmap_is_purged_on_open(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    store = HistoryStore(path)
    store.save(
        response(),
        submitted_text="tajni ulaz",
        store_text=False,
        mode="napredni",
        domain_hint=None,
    )
    with sqlite3.connect(path) as connection:
        old_response = response()
        old_response["explanations"] = {
            "token_heatmap": [{"text": "naslijeđeni tajni token"}],
            "top_features": [],
        }
        connection.execute(
            "UPDATE runs SET response_json = ? WHERE run_id = 'r1'",
            (json.dumps(old_response, ensure_ascii=False),),
        )

    reopened = HistoryStore(path)
    saved = reopened.get("r1")
    assert saved is not None
    assert saved["response"]["explanations"]["token_heatmap"] == []
    assert "naslijeđeni tajni token".encode() not in path.read_bytes()


def test_open_tolerates_null_response_in_legacy_started_row(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, submitted_text TEXT, response_json TEXT)"
        )
        connection.execute(
            "INSERT INTO runs (run_id, submitted_text, response_json) VALUES (?, ?, ?)",
            ("started", None, None),
        )

    HistoryStore(path)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT response_json FROM runs WHERE run_id = 'started'"
        ).fetchone()
    assert row == (None,)


_PRE_D267_RUNS = """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        verdict TEXT NOT NULL,
        p_ai REAL NOT NULL,
        interval_low REAL NOT NULL,
        interval_high REAL NOT NULL,
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
        runtime_identity_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE feedback (
        run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
        expected_label TEXT NOT NULL,
        comment TEXT,
        created_at TEXT NOT NULL
    );
"""


def test_history_written_with_the_pre_d267_interval_schema_still_opens(tmp_path: Path) -> None:
    """a history file whose NOT NULL interval columns hold the old ±0.10 band still
    lists and opens; the band is kept on disk but never returned, and new rows store NULL."""
    path = tmp_path / "history.sqlite3"
    old_response = {**response("old"), "interval": [0.6, 0.8], "p_ai": 0.7}
    with sqlite3.connect(path) as connection:
        connection.executescript(_PRE_D267_RUNS)
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old",
                "2026-10-01T10:00:00+00:00",
                "LIKELY_AI",
                0.7,
                0.6,
                0.8,
                "{}",
                "[]",
                "[]",
                "{}",
                json.dumps(old_response),
                None,
                "osnovni",
                "news",
                "succeeded",
                None,
                "{}",
            ),
        )
        connection.execute(
            "INSERT INTO feedback VALUES ('old', 'human', 'note', '2026-10-01T11:00:00+00:00')"
        )
    connection.close()

    store = HistoryStore(path)
    listed = store.list()
    assert [row["run_id"] for row in listed] == ["old"]
    assert not {"interval_low", "interval_high"} & set(listed[0])
    detail = store.get("old")
    assert detail is not None
    assert detail["verdict"] == "LIKELY_AI" and detail["p_ai"] == pytest.approx(0.7)
    assert not {"interval_low", "interval_high"} & set(detail)
    assert detail["feedback"]["expected_label"] == "human"
    assert "interval" not in store.export_csv()

    store.save(
        response("new"), submitted_text="tekst", store_text=False, mode="osnovni", domain_hint=None
    )
    assert {row["run_id"] for row in store.list()} == {"old", "new"}
    with sqlite3.connect(path) as connection:
        stored = dict(
            connection.execute("SELECT run_id, interval_low FROM runs ORDER BY run_id").fetchall()
        )
        notnull = {
            str(row[1]): int(row[3])
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
    connection.close()
    assert stored == {"new": None, "old": pytest.approx(0.6)}
    assert notnull["interval_low"] == 0 and notnull["interval_high"] == 0
