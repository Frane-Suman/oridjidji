"""`oridjidji serve` — M8's entry point, and its safety default (the specification, ETHICS.md).

The project plan has said "`serve` arrives in M8" since M0 and it never did. The command itself is
thin; what these tests pin is the part that is a decision rather than plumbing.

**Localhost by default.** `ETHICS.md` and the specification are explicit that this is a research and
educational tool that must never be used against an individual. A detector reachable from the
network is one somebody else can point at a student's essay. Exposing it has to be a
deliberate act, not the path of least resistance — so the default binds to 127.0.0.1 and any
other host prints a warning naming the ethics rule.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from oridjidji.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()


class _FakeUvicorn:
    """Records what `serve` would have run, without binding a socket."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, target: Any, **kwargs: Any) -> None:
        self.calls.append({"target": target, **kwargs})


@pytest.fixture
def uvicorn(monkeypatch: pytest.MonkeyPatch) -> _FakeUvicorn:
    fake = _FakeUvicorn()
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake)
    return fake


def test_serve_binds_localhost_by_default(uvicorn: _FakeUvicorn) -> None:
    """The safety default. A detector on 0.0.0.0 is one a third party can point at somebody's
    essay, and ETHICS.md forbids exactly that use."""
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert uvicorn.calls[0]["host"] == "127.0.0.1"


def test_binding_elsewhere_warns_and_names_the_rule(uvicorn: _FakeUvicorn) -> None:
    """Exposing it is allowed — this is the author's own machine — but it must not be
    silent. The warning quotes the rule rather than saying 'be careful'."""
    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 0
    assert "WARNING" in result.output
    assert "discipline" in result.output


def test_warmup_is_on_by_default(uvicorn: _FakeUvicorn) -> None:
    """Measured at 27.9 s. Without it the first user pays that and reads it as the
    tool being slow."""
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    # The app instance is constructed eagerly when not reloading, so warmup already ran.
    assert uvicorn.calls[0]["target"] is not None


def test_reload_passes_an_import_string_not_an_instance(uvicorn: _FakeUvicorn) -> None:
    """uvicorn's reloader re-imports the target, so it needs a string. Handing it a built
    app silently disables reloading."""
    result = runner.invoke(app, ["serve", "--reload"])
    assert result.exit_code == 0
    call = uvicorn.calls[0]
    assert isinstance(call["target"], str)
    assert call["factory"] is True


def test_a_missing_uvicorn_says_which_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """It lives in the `serve` extra so the library and its tests do not carry a web server.
    A bare ImportError would send the reader to the wrong problem."""
    import builtins

    real_import = builtins.__import__

    def _no_uvicorn(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "uvicorn":
            raise ModuleNotFoundError("No module named 'uvicorn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_uvicorn)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "--extra serve" in result.output
