"""llama.cpp Q4 backbone with exact token alignment and bounded reductions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from oridjidji.detectors.backbone import ReducedScores
from oridjidji.detectors.windowing import (
    WindowPolicy,
    combine_reduced,
    plan_word_aligned_windows,
    word_starts_from_offsets,
)

FloatArray = NDArray[np.float64]
LogitArray = NDArray[np.float32] | NDArray[np.float64]


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LlamaRuntime(Protocol):
    """Small boundary around llama-cpp-python, replaceable in zero-download tests."""

    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]: ...

    def logits(self, token_ids: Sequence[int]) -> LogitArray: ...

    def close(self) -> None: ...


class PythonLlamaRuntime:
    """Pinned llama-cpp-python adapter; `logits_all` is required by the formulas."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int,
        threads: int,
        n_gpu_layers: int = 0,
    ) -> None:
        from llama_cpp import Llama

        # 0 keeps every layer on the CPU, which is the laptop contract; -1 offloads all
        # of them, which is what makes the sealed Colab scoring run finish.
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
            embedding=False,
            verbose=False,
        )

    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
        return list(self._model.tokenize(text, add_bos=add_bos, special=special))

    def logits(self, token_ids: Sequence[int]) -> LogitArray:
        self._model.reset()  # type: ignore[no-untyped-call]
        self._model.eval(list(token_ids))
        rows = np.asarray(self._model._scores, dtype=np.float32)
        if rows.shape[0] != len(token_ids):
            raise RuntimeError("llama.cpp score buffer is misaligned with evaluated tokens")
        return rows

    def complete(self, prompt: str, *, max_tokens: int = 120) -> str:
        """Generate a short Croatian explanation with the already-loaded instruct model."""
        self._model.reset()  # type: ignore[no-untyped-call]
        response = cast(
            dict[str, Any],
            self._model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Tumači statističke rezultate na sažetom hrvatskom jeziku. "
                            "Ne mijenjaj izračun, ne tvrdi tko je autor i ne koristi popise. "
                            + prompt
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                seed=42,
            ),
        )
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("Gemma returned no interpretation choice")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Gemma returned an empty interpretation")
        return content.strip()

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if close is not None:
            close()


def verify_token_ids(reference: Sequence[int], actual: Sequence[int], probe: str) -> None:
    """A same-size vocabulary is insufficient; every input token must agree."""
    if list(reference) != list(actual):
        raise ValueError(f"GGUF and Hugging Face token IDs differ for Croatian probe {probe!r}")


def _log_normalizers(logits: LogitArray, chunk_size: int) -> FloatArray:
    maxima = np.asarray(np.max(logits, axis=1), dtype=np.float64)
    totals = np.zeros(len(logits), dtype=np.float64)
    for start in range(0, logits.shape[1], chunk_size):
        block = np.asarray(logits[:, start : start + chunk_size], dtype=np.float64)
        totals += np.exp(block - maxima[:, None]).sum(axis=1)
    return np.asarray(maxima + np.log(totals), dtype=np.float64)


def reduce_logits(
    observer_logits: LogitArray,
    performer_logits: LogitArray,
    targets: NDArray[np.int64],
    *,
    chunk_size: int = 8192,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Reduce two vocabulary distributions without allocating probability copies."""
    if observer_logits.shape != performer_logits.shape:
        raise ValueError("observer and performer GGUF logits have different shapes")
    if observer_logits.shape[0] != len(targets) or chunk_size < 1:
        raise ValueError("logit rows and realised targets are misaligned")
    observer_norm = _log_normalizers(observer_logits, chunk_size)
    performer_norm = _log_normalizers(performer_logits, chunk_size)
    realised = (
        np.asarray(observer_logits[np.arange(len(targets)), targets], dtype=np.float64)
        - observer_norm
    )
    expected = np.zeros(len(targets), dtype=np.float64)
    square = np.zeros(len(targets), dtype=np.float64)
    cross = np.zeros(len(targets), dtype=np.float64)
    maxima = np.zeros(len(targets), dtype=np.float64)
    below = np.zeros(len(targets), dtype=np.float64)
    for start in range(0, observer_logits.shape[1], chunk_size):
        observer = (
            np.asarray(observer_logits[:, start : start + chunk_size], dtype=np.float64)
            - observer_norm[:, None]
        )
        performer = (
            np.asarray(performer_logits[:, start : start + chunk_size], dtype=np.float64)
            - performer_norm[:, None]
        )
        probabilities = np.exp(observer)
        expected += (probabilities * observer).sum(axis=1)
        square += (probabilities * np.square(observer)).sum(axis=1)
        cross -= (probabilities * performer).sum(axis=1)
        maxima = np.maximum(maxima, probabilities.max(axis=1))
        below += np.where(observer <= realised[:, None], probabilities, 0.0).sum(axis=1)
    variance = np.maximum(square - np.square(expected), 0.0)
    return realised, expected, variance, cross, maxima, below


@dataclass(frozen=True)
class GgufPair:
    name: str
    observer_path: Path
    performer_path: Path
    observer_revision: str
    performer_revision: str
    tokenizer_hash: str
    llama_cpp_revision: str
    observer_sha256: str
    performer_sha256: str
    add_bos: bool
    quantization: str = "Q4_K_M"

    @property
    def identity(self) -> str:
        return (
            f"{self.name}:{self.observer_revision}/{self.performer_revision}:"
            f"{self.quantization}:llama.cpp@{self.llama_cpp_revision}"
        )

    def manifest(self, *, window_policy: WindowPolicy) -> dict[str, object]:
        return {
            "name": self.name,
            "observer_revision": self.observer_revision,
            "performer_revision": self.performer_revision,
            "observer_weights_sha256": self.observer_sha256,
            "performer_weights_sha256": self.performer_sha256,
            "tokenizer_mapping_sha256": self.tokenizer_hash,
            "backend": "llama.cpp",
            "precision": self.quantization,
            "quantization": self.quantization,
            "llama_cpp_revision": self.llama_cpp_revision,
            "add_bos": self.add_bos,
            "aggregation_version": "e9-v0-v10-v1",
            "window_policy": window_policy,
        }


class LlamaCppBackbone:
    """Two persistent GGUF models implementing the shared statistical evidence contract."""

    def __init__(
        self,
        pair: GgufPair,
        tokenizer: Any,
        *,
        threads: int = 8,
        window_policy: WindowPolicy = "prefix512",
        n_gpu_layers: int = 0,
        observer: LlamaRuntime | None = None,
        performer: LlamaRuntime | None = None,
    ) -> None:
        self.pair = pair
        self.tokenizer = tokenizer
        self.window_policy = window_policy
        self.threads = threads
        # Deliberately absent from `identity_manifest`: the weights, the tokenizer mapping and
        # the aggregation are identical wherever the kernels run.
        self.n_gpu_layers = n_gpu_layers
        self.observer = observer
        self.performer = performer

    @property
    def model_pair(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            self.identity_manifest(), sort_keys=True, separators=(",", ":")
        ).encode()
        return f"{self.pair.identity}:{hashlib.sha256(payload).hexdigest()[:16]}"

    def identity_manifest(self) -> dict[str, object]:
        """Complete serving identity used by caches, thresholds and bundles."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            binding_version = version("llama-cpp-python")
        except PackageNotFoundError:
            binding_version = "unavailable"
        manifest = self.pair.manifest(window_policy=self.window_policy)
        manifest["llama_cpp_python_version"] = binding_version
        return manifest

    def load(self) -> None:
        """Load once after verifying hashes, mapping, BOS handling and Croatian probes."""
        if self.observer is not None and self.performer is not None:
            return
        from oridjidji.detectors.hf_backbone import tokenizer_mapping_hash

        if tokenizer_mapping_hash(self.tokenizer) != self.pair.tokenizer_hash:
            raise ValueError("GGUF tokenizer mapping does not match the selected backbone")
        if _file_sha256(self.pair.observer_path) != self.pair.observer_sha256:
            raise ValueError("observer GGUF hash mismatch")
        if _file_sha256(self.pair.performer_path) != self.pair.performer_sha256:
            raise ValueError("performer GGUF hash mismatch")
        self.observer = PythonLlamaRuntime(
            self.pair.observer_path,
            n_ctx=1024,
            threads=self.threads,
            n_gpu_layers=self.n_gpu_layers,
        )
        self.performer = PythonLlamaRuntime(
            self.pair.performer_path,
            n_ctx=1024,
            threads=self.threads,
            n_gpu_layers=self.n_gpu_layers,
        )
        self._verify_probes()

    def _verify_probes(self) -> None:
        if self.observer is None or self.performer is None:
            raise ValueError("GGUF pair is not loaded")
        probes = ("Čovjek piše hrvatski tekst.", "„Riječ“ sadrži č, ć, đ, š i ž.")
        for probe in probes:
            reference = self.tokenizer.encode(probe, add_special_tokens=True)
            verify_token_ids(
                reference,
                self.observer.tokenize(
                    probe.encode("utf-8"), add_bos=self.pair.add_bos, special=True
                ),
                probe,
            )
            verify_token_ids(
                reference,
                self.performer.tokenize(
                    probe.encode("utf-8"), add_bos=self.pair.add_bos, special=True
                ),
                probe,
            )

    def _window(self, ids: list[int], starts: NDArray[np.bool_]) -> ReducedScores:
        if self.observer is None or self.performer is None:
            raise ValueError("GGUF pair is not loaded")
        observer = self.observer.logits(ids)[:-1]
        performer = self.performer.logits(ids)[:-1]
        targets = np.asarray(ids[1:], dtype=np.int64)
        realised, expected, variance, cross, maxima, pit = reduce_logits(
            observer, performer, targets
        )
        return ReducedScores(
            token_ids=targets,
            realised_logprobs=realised.astype(np.float32),
            expected_logprobs=expected.astype(np.float32),
            logprob_variances=variance.astype(np.float32),
            cross_entropies=cross.astype(np.float32),
            word_starts=starts,
            max_probs=maxima.astype(np.float32),
            pit=pit.astype(np.float32),
        )

    def score_tokens(self, text: str) -> ReducedScores:
        self.load()
        encoded = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        total = len(ids) - 1
        full_word_starts = word_starts_from_offsets(text, list(offsets[1:]))
        parts: list[ReducedScores] = []
        for window in plan_word_aligned_windows(total, full_word_starts, self.window_policy):
            stop = window.target_stop + 1
            flags = full_word_starts[window.context_start : window.target_stop]
            reduced = self._window(ids[window.context_start : stop], flags)
            drop = window.context_tokens
            parts.append(_slice(reduced, drop))
        return combine_reduced(parts, total_tokens=total, policy=self.window_policy)

    def close(self) -> None:
        if self.observer is not None:
            self.observer.close()
        if self.performer is not None:
            self.performer.close()
        self.observer = None
        self.performer = None


def _slice(scores: ReducedScores, start: int) -> ReducedScores:
    values = slice(start, None)
    return ReducedScores(
        token_ids=scores.token_ids[values],
        realised_logprobs=scores.realised_logprobs[values],
        expected_logprobs=scores.expected_logprobs[values],
        logprob_variances=scores.logprob_variances[values],
        cross_entropies=scores.cross_entropies[values]
        if scores.cross_entropies is not None
        else None,
        word_starts=scores.word_starts[values] if scores.word_starts is not None else None,
        max_probs=scores.max_probs[values] if scores.max_probs is not None else None,
        pit=scores.pit[values] if scores.pit is not None else None,
    )
