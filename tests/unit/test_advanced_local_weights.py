"""Locally quantised Gemma weights cannot reproduce the packaged hashes."""

from pathlib import Path

import pytest

from oridjidji.fusion.advanced_bundle import _locally_built

GGUF = Path("models/statistical/observer.Q4_K_M.gguf")


def test_locally_built_weights_are_refused_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORIDJIDJI_SKIP_BUNDLE_VERIFY", raising=False)
    assert not _locally_built(GGUF)


def test_operator_flag_accepts_only_the_statistical_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORIDJIDJI_SKIP_BUNDLE_VERIFY", "1")
    assert _locally_built(GGUF)
    assert _locally_built(Path("models/statistical/gguf_pair_manifest.json"))
    assert not _locally_built(Path("models/basic/model/model.onnx"))
    assert not _locally_built(Path("models/advanced/calibration/fusion.layer.json"))
