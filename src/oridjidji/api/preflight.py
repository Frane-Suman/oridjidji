"""Native localhost preflight for the Basic and Advanced demo modes."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field

from oridjidji.api.schemas import ConfigResponse, HealthResponse, RuntimeCapabilities


class SmokeResult(BaseModel):
    """One optional real-model document request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status_code: int
    runtime_status: str | None = None
    agent_names: list[str] = Field(default_factory=list)
    error: str | None = None


class DemoPreflight(BaseModel):
    """Machine-readable result of checking the local demo contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str
    ok: bool
    health_status: str
    modes: list[str]
    capabilities_by_mode: dict[str, RuntimeCapabilities]
    smoke: dict[str, SmokeResult] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


def _post(client: TestClient, mode: str, text: str) -> SmokeResult:
    response = client.post("/api/v1/analyze", json={"text": text, "mode": mode})
    if response.status_code != 200:
        return SmokeResult(status_code=response.status_code, error=response.text[:500])
    body = response.json()
    return SmokeResult(
        status_code=response.status_code,
        runtime_status=body["runtime_identity"]["runtime_status"],
        agent_names=[str(agent["name"]) for agent in body["agents"]],
    )


def inspect_local_demo(
    profile: str, *, smoke_text: str | None = None, require_advanced: bool = True
) -> DemoPreflight:
    """Verify the live API contract and optionally score one document in each mode.

    ``research_only`` is accepted for Advanced: it qualifies the evidence and never
    disables the mode.  The check requires the A1+A2+A3 document path, while segment
    reliability remains an independently reported optional capability. With
    ``require_advanced=False`` only the Basic path is required, for installations
    where the optional Advanced layer has not been prepared.
    """
    from oridjidji.api.app import create_app

    previous = os.environ.get("ORIDJIDJI_PROFILE")
    os.environ["ORIDJIDJI_PROFILE"] = profile
    try:
        application = create_app(warmup=False)
        with TestClient(application) as client:
            health = HealthResponse.model_validate(client.get("/api/v1/health").json())
            config = ConfigResponse.model_validate(client.get("/api/v1/config").json())
            route_methods: set[tuple[str, str]] = set()
            for route in application.routes:
                path = getattr(route, "path", None)
                if not isinstance(path, str):
                    continue
                for method in getattr(route, "methods", None) or set():
                    route_methods.add((path, str(method)))
            errors: list[str] = []
            expected_modes = ["osnovni", "napredni"]
            if require_advanced and list(config.modes) != expected_modes:
                errors.append("API does not expose both Basic and Advanced modes")
            for path in ("/api/v1/analyze", "/api/v1/analyze/stream"):
                if (path, "POST") not in route_methods:
                    errors.append(f"missing POST route: {path}")
            basic = config.capabilities_by_mode.get("osnovni", RuntimeCapabilities())
            advanced = config.capabilities_by_mode.get("napredni", RuntimeCapabilities())
            if not basic.document_detection:
                errors.append("Basic calibrated document detection is unavailable")
            if require_advanced and not advanced.document_detection:
                errors.append("Advanced fused document detection is unavailable")
            if require_advanced and not advanced.statistical_evidence:
                errors.append("Advanced A2/A3 statistical evidence is unavailable")

            smoke: dict[str, SmokeResult] = {}
            if smoke_text is not None:
                smoke = {
                    "osnovni": _post(client, "osnovni", smoke_text),
                }
                if require_advanced:
                    smoke["napredni"] = _post(client, "napredni", smoke_text)
                for mode, result in smoke.items():
                    if result.status_code != 200:
                        errors.append(f"{mode} smoke request failed: {result.error}")
                advanced_smoke = smoke.get("napredni")
                if advanced_smoke is not None and advanced_smoke.runtime_status == "degraded_basic":
                    errors.append("Advanced smoke fell back to Basic")

            return DemoPreflight(
                profile=profile,
                ok=not errors,
                health_status=health.status,
                modes=[str(mode) for mode in config.modes],
                capabilities_by_mode={
                    mode: capabilities for mode, capabilities in config.capabilities_by_mode.items()
                },
                smoke=smoke,
                errors=errors,
            )
    finally:
        if previous is None:
            os.environ.pop("ORIDJIDJI_PROFILE", None)
        else:
            os.environ["ORIDJIDJI_PROFILE"] = previous


def read_smoke_text(path: Path) -> str:
    """Read the explicitly supplied UTF-8 smoke document."""
    return path.read_text(encoding="utf-8")
