"""Exercise the real Basic and Advanced candidates through the browser workbench."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import uvicorn
from playwright.sync_api import Page, expect, sync_playwright

from oridjidji.api.app import create_app
from oridjidji.storage import HistoryStore

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
PROFILE = "e12-advanced-research"
SMOKE_TEXT = (
    ROOT
    / "data"
    / "processed"
    / "models"
    / "candidates"
    / "e12-advanced-research"
    / "demo_smoke_hr.txt"
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_server(base_url: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/api/v1/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError("real frontend smoke server did not start")


def _submit(page: Page, base_url: str, *, mode: str, text: str) -> dict[str, object]:
    page.goto(base_url)
    selector = "mode-advanced" if mode == "napredni" else "mode-basic"
    page.get_by_test_id(selector).check()
    page.get_by_test_id("editor").fill(text)
    started = time.perf_counter()
    page.get_by_test_id("submit").click()
    expect(page.get_by_test_id("result-heading")).to_be_focused(timeout=180_000)
    elapsed = time.perf_counter() - started
    response = page.evaluate("window.__oridjidjiLastResult")
    if not isinstance(response, dict):
        raise RuntimeError(f"{mode} browser response was not captured")
    if response.get("requested_mode") != mode or response.get("effective_mode") != mode:
        raise RuntimeError(f"{mode} did not execute as requested: {response}")
    runtime = response["runtime_identity"]
    return {
        "requested_mode": response["requested_mode"],
        "effective_mode": response["effective_mode"],
        "runtime_status": runtime["runtime_status"],
        "run_id": response["run_id"],
        "agent_names": [row["name"] for row in response.get("agent_evidence", [])],
        "assessment_agents_used": response["assessment_agents_used"],
        "segment_available": runtime["capabilities"]["segment_reliability"],
        "browser_elapsed_seconds": round(elapsed, 3),
    }


def main() -> None:
    """Start the actual profile, submit both modes in Edge and save the evidence."""
    if not EDGE.is_file():
        raise FileNotFoundError(f"Edge browser is unavailable: {EDGE}")
    if not SMOKE_TEXT.is_file():
        raise FileNotFoundError(f"Croatian smoke document is unavailable: {SMOKE_TEXT}")
    os.environ["ORIDJIDJI_PROFILE"] = PROFILE
    text = SMOKE_TEXT.read_text(encoding="utf-8")
    # llama.cpp may keep a Windows file handle alive briefly while the server thread
    # releases its models. The evidence is already complete at that point, so cleanup
    # retries must not turn a successful browser smoke into a false failure.
    with TemporaryDirectory(
        prefix="oridjidji-frontend-smoke-", ignore_cleanup_errors=True
    ) as temporary:
        history = HistoryStore(Path(temporary) / "history.sqlite3")
        application = create_app(warmup=True, history=history)
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(
            uvicorn.Config(application, host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            _wait_for_server(base_url)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=str(EDGE), headless=True)
                context = browser.new_context(locale="hr-HR")
                context.add_init_script(
                    "window.addEventListener('oridjidji:result', "
                    "event => { window.__oridjidjiLastResult = event.detail; });"
                )
                page = context.new_page()
                page.set_default_timeout(180_000)
                basic = _submit(page, base_url, mode="osnovni", text=text)
                advanced = _submit(page, base_url, mode="napredni", text=text)
                browser.close()
            history_payload = httpx.get(f"{base_url}/api/v1/runs", timeout=5).json()
            history_rows = (
                history_payload["items"] if isinstance(history_payload, dict) else history_payload
            )
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    if {row["mode"] for row in history_rows} != {"osnovni", "napredni"}:
        raise RuntimeError("real browser smoke did not persist both execution modes")
    report = {
        "version": "m8-real-browser-smoke-v1",
        "profile": PROFILE,
        "browser": str(EDGE),
        "smoke_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "storage_requested": False,
        "basic": basic,
        "advanced": advanced,
        "history_modes": sorted(row["mode"] for row in history_rows),
    }
    output = ROOT / "docs" / "results" / "m8_frontend_real_browser_smoke.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
