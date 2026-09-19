"""A real localhost server and the installed Edge browser for frontend tests."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from playwright.sync_api import Browser, Playwright, sync_playwright

from oridjidji.api.app import create_app
from oridjidji.storage import HistoryStore

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


@pytest.fixture(scope="session")
def live_site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve the real FastAPI/Jinja/Alpine stack with model doubles (no weights)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    history = HistoryStore(tmp_path_factory.mktemp("browser-history") / "history.sqlite3")
    application = create_app(detectors=[], history=history)
    config = uvicorn.Config(application, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/api/v1/health", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=3)
        raise RuntimeError("browser-test server did not start")
    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def browser() -> Iterator[Browser]:
    """Use one Playwright loop per test so later asyncio unit tests see no running loop."""
    if not EDGE.is_file():
        pytest.skip("system Edge is unavailable")
    playwright: Playwright = sync_playwright().start()
    instance = playwright.chromium.launch(executable_path=str(EDGE), headless=True)
    yield instance
    instance.close()
    playwright.stop()
