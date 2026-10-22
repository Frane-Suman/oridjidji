"""The web layer.

Most of what can go wrong here is not a crash. It is a screen that renders correctly and
claims slightly more than the model can support — a missing notice, a headline that reads
as clearance, a hard-coded threshold that survives a recalibration. Those are what these
tests are aimed at.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from oridjidji.api.app import (
    AdvancedAdmission,
    AdvancedBusy,
    RequestRateLimiter,
    create_app,
)
from oridjidji.i18n import MESSAGES_HR, NOTICE_HR
from oridjidji.storage import HistoryStore
from oridjidji.web import render

pytestmark = pytest.mark.unit


@pytest.fixture
def client() -> TestClient:
    """No detectors, deliberately.

    `create_app()` discovers whatever sits in `data/processed/models`, so once the M4
    checkpoint landed these tests silently began running real ONNX inference: the suite went
    from 22 s to 100 s and started depending on weights that are not in git. Unit tests mock
    the model layer, and this file is about the web layer, which holds no model
    facts by construction. Detector behaviour is covered in `test_detector_serving.py`
    against a stub.
    """
    return TestClient(create_app([]))


CROATIAN = (
    "Gradsko vijeće raspravljalo je o proračunu za sljedeću godinu i o ulaganjima u "
    "komunalnu infrastrukturu diljem grada. "
) * 20


# ----------------------------------------------------------------- formatting


def test_numbers_use_the_croatian_decimal_comma() -> None:
    """`78.4` next to Croatian prose is wrong typography, and it reads as a foreign tool."""
    assert render.hr_percent(0.784) == "78,4"
    assert render.hr_number(64.06, decimals=1) == "64,1"
    # Ties round half-to-even, as Python formats them. Noted rather than worked around:
    # a displayed percentage is not a measurement whose last digit anyone should read.
    assert render.hr_number(64.05, decimals=1) == "64,0"


def test_the_result_never_shows_the_heuristic_range() -> None:
    """the ±0.10 display band was not a statistically grounded
    uncertainty estimate, so no part of the displayed result may state or draw it."""
    ctx = render.result_context(
        verdict="LIKELY_AI",
        p_ai=0.784,
        n_words=312,
        agents_used=3,
        agents_total=3,
        limitations=["x"],
    )
    assert not {"interval_line", "low", "high", "low_pct", "high_pct"} & set(ctx)
    shown = " ".join(str(value) for value in ctx.values())
    assert "64,1" not in shown and "88,2" not in shown
    assert "raspon" not in str(ctx["rail_label"]).lower()


# ----------------------------------------------------------------- the claim made


def test_no_headline_ever_claims_the_text_is_human() -> None:
    """The decision recorded in `i18n.py`: every headline reports a degree of MACHINE
    signal or its absence, never a person.

    A result reading "probably human" is the output most likely to be quoted at someone
    as clearance, and the model cannot support it. The specification forbids any verified-human state.
    """
    headlines = [render.headline_for(v) for v in render.ZONE_FOR_VERDICT]
    for headline in headlines:
        lowered = headline.lower()
        assert "ljudsk" not in lowered, f"{headline!r} claims humanity"
        assert "čovjek" not in lowered
    assert render.headline_for("LIKELY_HUMAN") == "Slabi znakovi strojnog pisanja"


def test_an_unknown_verdict_fails_loudly_rather_than_defaulting() -> None:
    """A silent fallback would render an unrecognised verdict in the calmest-looking
    zone, which is the wrong direction to guess in."""
    with pytest.raises(ValueError, match="Unknown verdict"):
        render.zone_for("DEFINITELY_AI")


def test_every_verdict_maps_to_a_zone() -> None:
    assert set(render.ZONE_FOR_VERDICT) == {"LIKELY_HUMAN", "INCONCLUSIVE", "LIKELY_AI"}
    assert set(render.ZONE_FOR_VERDICT.values()) == {"low", "inconclusive", "high"}


# ----------------------------------------------------------------- the rail


def test_the_rail_marks_only_the_estimate() -> None:
    """The rail places the calibrated estimate on a neutral 0-100 scale and draws no band; the three-state verdict carries the uncertainty, not a numeric interval."""
    ctx = render.result_context(
        verdict="LIKELY_AI",
        p_ai=0.784,
        n_words=312,
        agents_used=3,
        agents_total=3,
        limitations=["x"],
    )
    assert ctx["estimate_pct"] == pytest.approx(78.4)
    assert "low_pct" not in ctx and "high_pct" not in ctx


# ----------------------------------------------------------------- the ring


def test_the_ring_has_one_band_per_detector_reported_by_the_api() -> None:
    """Three detectors, three bands; five, five. Geometry is never hard-coded."""
    for count in (1, 3, 5):
        radii = {
            round(((d.cx - 200) ** 2 + (d.cy - 200) ** 2) ** 0.5, 1)
            for i in range(count)
            for d in render.ring_dots(i, count)
        }
        assert len(radii) == count, f"{count} bands should sit at {count} distinct radii"


def test_ring_dots_start_empty_not_partially_filled() -> None:
    """A ring that starts half full implies progress that has not happened."""
    for dot in render.ring_dots(0, 3):
        assert dot.r < 2.0
        assert dot.opacity < 0.25


def test_no_bands_when_the_api_reports_no_detectors() -> None:
    """Honest: before M4 nothing is loaded, so the ring renders nothing rather than
    placeholder bands implying detectors that do not exist."""
    assert render.ring_dots(0, 0) == []


# ----------------------------------------------------------------- the notice


def test_the_notice_carries_all_three_sentences() -> None:
    assert len(NOTICE_HR) == 3
    assert NOTICE_HR[0] == "Ovo je statistička procjena, a ne dokaz."


def test_every_result_state_renders_the_full_notice(client: TestClient) -> None:
    """The specification: the notice is visible in every result screenshot, without scrolling. It is
    furniture, so no result state may omit it — including the ones nobody looks at."""
    html = client.get("/pregled").text
    for sentence in NOTICE_HR:
        assert html.count(sentence) == 6, f"{sentence!r} missing from a result state"


# ----------------------------------------------------------------- routes


def test_the_editor_occupies_the_workbench_before_the_first_run(client: TestClient) -> None:
    """The initial workbench is the editor; the result replaces it after submission."""
    html = client.get("/").text
    assert 'data-testid="editor-pane"' in html
    assert "x-show=\"state !== 'idle'\"" in html


def test_the_review_page_shows_all_three_states(client: TestClient) -> None:
    """The specification requires the three result states screenshotted side by side, and the stub
    only ever returns INCONCLUSIVE — so this route is how the comparison is possible."""
    html = client.get("/pregled").text
    for verdict in render.ZONE_FOR_VERDICT:
        assert render.headline_for(verdict) in html


def test_the_review_page_shows_runtime_qualification_states(client: TestClient) -> None:
    html = client.get("/pregled").text
    for key in (
        "ui_runtime_research_title",
        "ui_runtime_fallback_title",
        "ui_runtime_unavailable_title",
    ):
        assert MESSAGES_HR[key] in html


def test_the_review_page_says_its_numbers_are_invented(client: TestClient) -> None:
    """A page rendering fabricated results must never be mistakable for a real analysis."""
    assert "RAZVOJNI PREGLED" in client.get("/pregled").text


def test_config_serves_the_limits_the_frontend_is_not_allowed_to_know(
    client: TestClient,
) -> None:
    """The specification: no length limits in the frontend. They have to be servable to be absent
    from it."""
    cfg = client.get("/api/v1/config").json()
    assert cfg["min_words"] > 0
    assert cfg["max_words"] > cfg["min_words"]
    assert "agent_labels_hr" in cfg
    assert cfg["modes"] == ["osnovni", "napredni"]
    assert set(cfg["capabilities_by_mode"]) == {"osnovni", "napredni"}
    assert set(cfg["capabilities_by_mode"]["napredni"]) == {
        "document_detection",
        "statistical_evidence",
    }


# ----------------------------------------------------------------- the stream


def test_the_stream_reports_steps_then_a_result(client: TestClient) -> None:
    with client.stream("POST", "/api/v1/analyze/stream", json={"text": CROATIAN}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line for line in response.iter_lines() if line.startswith("event:")]
    assert events[0] == "event: step"
    assert events[-1] == "event: result"


def test_stream_uses_the_same_post_body_contract_as_analyze(client: TestClient) -> None:
    assert client.get("/api/v1/analyze/stream", params={"text": CROATIAN}).status_code == 405
    with client.stream(
        "POST",
        "/api/v1/analyze/stream",
        json={"text": CROATIAN, "mode": "osnovni", "domain_hint": "news", "store": False},
    ) as response:
        assert response.status_code == 200


def test_post_and_stream_return_the_same_final_assessment(client: TestClient) -> None:
    import json

    direct = client.post("/api/v1/analyze", json={"text": CROATIAN, "mode": "napredni"}).json()
    with client.stream(
        "POST", "/api/v1/analyze/stream", json={"text": CROATIAN, "mode": "napredni"}
    ) as response:
        result_payload = next(
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and '"response"' in line
        )["response"]
    for key in (
        "verdict",
        "p_ai",
        "operating_decision_ai",
        "prediction_set",
        "agents",
        "explanations",
        "warnings",
        "runtime_identity",
    ):
        assert result_payload[key] == direct[key]
    assert result_payload["gates"] | {"language_confidence": None} == direct["gates"] | {
        "language_confidence": None
    }
    assert result_payload["gates"]["language_confidence"] == pytest.approx(
        direct["gates"]["language_confidence"]
    )


def test_stream_has_no_removed_segment_phase(client: TestClient) -> None:
    with client.stream(
        "POST", "/api/v1/analyze/stream", json={"text": CROATIAN, "mode": "napredni"}
    ) as response:
        events = [
            line.removeprefix("event: ")
            for line in response.iter_lines()
            if line.startswith("event:")
        ]
    assert "segment_started" not in events
    assert "segment_finished" not in events
    assert events.index("run_finished") < events.index("result")
    assert events[-1] == "result"


def test_request_types_and_body_limit_are_enforced(client: TestClient) -> None:
    assert (
        client.post("/api/v1/analyze", json={"text": CROATIAN, "mode": "unknown"}).status_code
        == 422
    )
    assert (
        client.post("/api/v1/analyze", json={"text": CROATIAN, "domain_hint": "sports"}).status_code
        == 422
    )
    oversized = "a" * (128 * 1024)
    assert client.post("/api/v1/analyze", json={"text": oversized}).status_code == 413


def test_cors_is_limited_to_local_browser_origins(client: TestClient) -> None:
    local = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "POST",
        },
    )
    remote = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert local.headers["access-control-allow-origin"] == "http://localhost:8000"
    assert "access-control-allow-origin" not in remote.headers


def test_health_reports_actual_memory_and_typed_runtime_identity(client: TestClient) -> None:
    health = client.get("/api/v1/health").json()
    assert health["rss_mb"] > 0
    assert set(health["runtime_identity"]) == {
        "encoder",
        "causal_pair",
        "fusion",
        "calibration",
        "conformal",
        "repository",
        "preprocessing",
        "runtime_status",
        "device_transfer_status",
        "probability_device",
        "limitations",
        "capabilities",
    }
    assert set(health["runtime_identity"]["capabilities"]) == {
        "document_detection",
        "statistical_evidence",
    }


def test_advanced_admission_allows_one_active_and_one_waiting() -> None:
    async def exercise() -> None:
        admission = AdvancedAdmission()
        await admission.acquire()
        waiting = asyncio.create_task(admission.acquire())
        await asyncio.sleep(0)
        with pytest.raises(AdvancedBusy):
            await admission.acquire()
        await admission.release()
        await waiting
        await admission.release()

    asyncio.run(exercise())


def test_analysis_rate_limiter_returns_retry_after() -> None:
    async def exercise() -> None:
        limiter = RequestRateLimiter(1)
        await limiter.check("local")
        with pytest.raises(Exception) as caught:
            await limiter.check("local")
        assert getattr(caught.value, "status_code", None) == 429
        assert int(caught.value.headers["Retry-After"]) >= 1

    asyncio.run(exercise())


def test_refused_analysis_is_persisted_without_text(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = HistoryStore(tmp_path / "history.sqlite3")
    local = TestClient(create_app([], history=store))
    assert local.post("/api/v1/analyze", json={"text": "Prekratko."}).status_code == 422
    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["status"] == "refused"
    assert rows[0]["text_stored"] is False


def test_a_refusal_streams_an_error_not_a_result(client: TestClient) -> None:
    """ "Too short" has no defensible answer, so the stream must not end in a result
    . A UI that received one would render a number for a text it cannot score."""
    with client.stream("POST", "/api/v1/analyze/stream", json={"text": "Prekratko."}) as response:
        events = [line for line in response.iter_lines() if line.startswith("event:")]
    assert events[-1] == "event: error"
    assert "event: result" not in events


def test_every_step_label_travels_in_the_event(client: TestClient) -> None:
    """The specification: the frontend never invents a step label, so every event has to carry one."""
    import json

    with client.stream("POST", "/api/v1/analyze/stream", json={"text": CROATIAN}) as response:
        payloads = [
            json.loads(line.removeprefix("data:"))
            for line in response.iter_lines()
            if line.startswith("data:")
        ]
    for payload in payloads:
        if "step_hr" in payload:
            assert payload["step_hr"].strip(), "an event carried an empty label"


# ----------------------------------------------------------------- the specification consistency


def test_no_model_facts_are_hard_coded_in_the_frontend() -> None:
    """The mechanical proof of the standing rule.

    Thresholds, detector names and step labels in the templates or JS would couple the
    UI to a model version and break silently the moment calibration is re-run.
    """
    import re
    from pathlib import Path

    web = Path(render.__file__).parent
    sources = list((web / "templates").rglob("*.html")) + [web / "static" / "js" / "app.js"]
    forbidden = ("binoculars", "fast-detectgpt", "fast_detect", "0.5", "threshold", "prag =")

    # Comments are stripped first. Several of these files explain in prose WHY they hold
    # no thresholds or detector names, and a scan that cannot tell an explanation from a
    # violation trains you to ignore it.
    for path in sources:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\{#.*?#\}", "", text, flags=re.S)  # Jinja comments
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # JS block comments
        text = re.sub(r"^\s*//.*$", "", text, flags=re.M)  # JS line comments
        lowered = text.lower()
        for needle in forbidden:
            assert needle not in lowered, f"{path.name} hard-codes {needle!r}"


def test_component_css_holds_no_literal_hex_or_raw_ms() -> None:
    """Every value that is not a token escaped the design system."""
    import re
    from pathlib import Path

    css = (Path(render.__file__).parent / "static" / "css" / "app.css").read_text(encoding="utf-8")
    without_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", without_comments)
    assert not re.search(r"\b\d+ms\b", without_comments)
    assert "backdrop-filter: blur(var(--masthead-blur))" in without_comments
    assert "box-shadow" not in without_comments
