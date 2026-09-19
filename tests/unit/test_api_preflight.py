from __future__ import annotations

from fastapi import FastAPI

from oridjidji.api import app as app_module
from oridjidji.api.preflight import inspect_local_demo


def _contract_app(*, advanced_status: str = "research_only") -> FastAPI:
    app = FastAPI()
    capabilities = {
        "osnovni": {
            "document_detection": True,
            "statistical_evidence": False,
        },
        "napredni": {
            "document_detection": True,
            "statistical_evidence": True,
        },
    }

    @app.get("/api/v1/health")
    def health() -> dict[str, object]:
        return {
            "status": "research",
            "version": "test",
            "runtime_identity": {
                "runtime_status": "research_only",
                "capabilities": capabilities["napredni"],
            },
        }

    @app.get("/api/v1/config")
    def config() -> dict[str, object]:
        return {
            "modes": ["osnovni", "napredni"],
            "min_words": 100,
            "max_words": 5000,
            "capabilities_by_mode": capabilities,
            "disclaimer_hr": "Istraživački prototip.",
            "limitations_hr": ["Ograničenje."],
        }

    def result(mode: str) -> dict[str, object]:
        return {
            "runtime_identity": {
                "runtime_status": advanced_status if mode == "napredni" else "basic"
            },
            "agents": [{"name": "encoder"}],
        }

    @app.post("/api/v1/analyze")
    def analyze(payload: dict[str, object]) -> dict[str, object]:
        return result(str(payload["mode"]))

    @app.post("/api/v1/analyze/stream")
    def stream() -> dict[str, str]:
        return {"status": "available"}

    return app


def test_research_only_advanced_is_available(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(app_module, "create_app", lambda **_kwargs: _contract_app())
    result = inspect_local_demo("test")
    assert result.ok
    assert result.modes == ["osnovni", "napredni"]
    assert result.capabilities_by_mode["napredni"].document_detection
    assert result.capabilities_by_mode["napredni"].statistical_evidence


def test_smoke_rejects_an_advanced_request_that_only_fell_back_to_basic(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(
        app_module,
        "create_app",
        lambda **_kwargs: _contract_app(advanced_status="degraded_basic"),
    )
    result = inspect_local_demo("test", smoke_text="Dovoljno dug hrvatski tekst.")
    assert not result.ok
    assert "Advanced smoke fell back to Basic" in result.errors
