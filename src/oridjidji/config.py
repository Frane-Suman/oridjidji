"""Configuration loading: configs/default.yaml overlaid by a profile file.

The ``lite`` profile is the default, not a fallback: the target machine has no
discrete GPU, so all inference is CPU-only with quantised
models. ``full`` exists for machines with more headroom and is still CPU-only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(
    os.environ.get("ORIDJIDJI_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs")
)

_CANDIDATE_PATH_KEYS = frozenset(
    {
        "model_dir",
        "calibration_dir",
        "encoder_candidate_manifest",
        "advanced_candidate_manifest",
        "statistical_pair_manifest",
        "statistical_tokenizer",
        "linguistic_dir",
        "fusion_dir",
        "bundle_path",
        "history_db",
    }
)


class AppConfig(BaseModel):
    """Application configuration after the profile overlay."""

    profile: str
    language: str = "hr"
    seed: int = 42
    # The specification's gate. Derived from the length curve, not asserted and the
    # comment in configs/default.yaml, which carries the measurement behind the value.
    min_word_count: int = 100
    inference: dict[str, Any] = Field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _resolve_candidate_paths(inference: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve only declared artifact paths from one portable candidate root."""
    resolved = dict(inference)
    candidate_root = root.resolve()
    for key in _CANDIDATE_PATH_KEYS:
        value = resolved.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            resolved[key] = str((candidate_root / path).resolve())
    return resolved


def load_config(
    profile: str = "lite",
    config_dir: Path | None = None,
    candidate_root: Path | None = None,
) -> AppConfig:
    """Read ``default.yaml``, overlay ``profiles/<profile>.yaml``, validate.

    The overlay is a one-level merge: profile keys win; nested mappings are merged
    one level deep. Deep-merge machinery is deliberately avoided — one level is
    all this project needs and anything cleverer costs explanation budget.
    """
    base_dir = config_dir or CONFIG_DIR
    merged = _read_yaml(base_dir / "default.yaml")
    overlay = _read_yaml(base_dir / "profiles" / f"{profile}.yaml")
    for key, value in overlay.items():
        base_value = merged.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            merged[key] = {**base_value, **value}
        else:
            merged[key] = value
    merged["profile"] = profile
    configured_root = candidate_root
    if configured_root is None:
        value = os.environ.get("ORIDJIDJI_CANDIDATE_ROOT")
        configured_root = Path(value) if value else None
    inference = merged.get("inference")
    if configured_root is not None and isinstance(inference, dict):
        merged["inference"] = _resolve_candidate_paths(inference, configured_root)
    return AppConfig(**merged)


class Secrets(BaseSettings):
    """Optional API keys from the environment / ``.env``.

    Every key is optional: the system must work fully with local models alone.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    mistral_api_key: str | None = None
    openrouter_api_key: str | None = None
    deepseek_api_key: str | None = None
    hf_token: str | None = None
