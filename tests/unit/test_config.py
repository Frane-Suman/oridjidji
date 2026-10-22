import importlib
from pathlib import Path

import pytest

import oridjidji.config as config_module
from oridjidji.config import load_config

pytestmark = pytest.mark.unit


def test_lite_profile_loads_with_defaults() -> None:
    cfg = load_config("lite")
    assert cfg.profile == "lite"
    assert cfg.language == "hr"
    assert cfg.min_word_count == 100  # the shipped product gate
    assert cfg.inference["device"] == "cpu"
    assert cfg.inference["encoder_backend"] == "onnx"


def test_full_profile_overlays_but_keeps_base_keys() -> None:
    cfg = load_config("full")
    assert cfg.inference["encoder_backend"] == "torch-fp32"
    assert cfg.inference["device"] == "cpu"  # one-level nested merge keeps base keys


def test_unknown_profile_raises_clear_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist")


def test_portable_advanced_profile_uses_mounted_candidate_paths() -> None:
    cfg = load_config("demo")
    assert cfg.inference["advanced_candidate_manifest"] == "advanced_candidate_manifest.json"
    assert cfg.inference["encoder_candidate_manifest"] == ("basic/encoder_candidate_manifest.json")
    assert cfg.inference["statistical_pair_manifest"] == ("statistical/gguf_pair_manifest.json")
    assert cfg.inference["history_db"] == "history/history.sqlite3"


def test_portable_profile_resolves_declared_paths_from_candidate_root(tmp_path: Path) -> None:
    cfg = load_config("demo", candidate_root=tmp_path)

    assert cfg.inference["advanced_candidate_manifest"] == str(
        (tmp_path / "advanced_candidate_manifest.json").resolve()
    )
    assert cfg.inference["encoder_candidate_manifest"] == str(
        (tmp_path / "basic" / "encoder_candidate_manifest.json").resolve()
    )
    assert cfg.inference["backbone"] == "gemma2_2b"


def test_candidate_root_environment_supports_installed_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORIDJIDJI_CANDIDATE_ROOT", str(tmp_path))

    cfg = load_config("demo")

    assert Path(cfg.inference["statistical_pair_manifest"]).is_relative_to(tmp_path)


def test_installed_runtime_can_take_config_root_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORIDJIDJI_CONFIG_DIR", str(tmp_path))
    reloaded = importlib.reload(config_module)
    try:
        assert tmp_path == reloaded.CONFIG_DIR
    finally:
        monkeypatch.delenv("ORIDJIDJI_CONFIG_DIR")
        importlib.reload(config_module)
