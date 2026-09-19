"""The concrete observer/performer pair, backed by transformers.

`backbone.py` owns the contract; this owns the weights. Two causal LMs are loaded once,
share one tokenizer, and every detector reads their output through `Backbone.score_tokens`
— which is the whole point of the specification, because Binoculars needs both models and
Fast-DetectGPT needs one, and three detectors each loading their own copy is how a 16 GB
laptop stops being able to run the Advanced mode at all.

**The vocabulary axis is reduced inside the forward pass and never returned.** Full
distributions at the 160-token serving window are 82 MB per model for EuroLLM's 128k
vocabulary and 97 MB for Qwen3's 152k; the four scalars per token that the detectors
actually consume are about 3 KB. `score_tokens` returns `ReducedScores` for that reason,
and `score_tokens_full` exists for E4's grid, which does want the distributions.

Imports of `torch` and `transformers` are function-local: the fast suite installs neither
, and every test here runs on constructed arrays rather than weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from oridjidji.detectors.backbone import ReducedScores, TokenScores
from oridjidji.detectors.windowing import (
    WindowPolicy,
    combine_reduced,
    plan_word_aligned_windows,
    word_starts_from_offsets,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

# The window the encoder already serves at, for the same reason: the specification's latency budget and
# the length curve meet at 160 tokens. Scoring a wider window than the detector
# ships with would report a number the served system never produces.
DEFAULT_MAX_TOKENS = 160

# Below this, the statistic is noise rather than a measurement — Fast-DetectGPT's own
# floor, restated here because the backbone is where a short text gets rejected.
MIN_TOKENS = 32


@dataclass(frozen=True)
class BackbonePair:
    """One observer/performer pair, named as it will appear in every threshold artefact.

    `observer_id` and `performer_id` are Hugging Face ids, pinned to a revision where the
    hub offers one. An earlier decision is why they are recorded rather than described: a model card's
    language list is a claim, and `aya-101` entered a roster on reputation while not
    supporting Croatian at all.
    """

    name: str
    observer_id: str
    performer_id: str
    observer_revision: str | None = None
    performer_revision: str | None = None
    observer_weights_sha256: str | None = None
    performer_weights_sha256: str | None = None
    tokenizer_mapping_sha256: str | None = None

    @property
    def model_pair(self) -> str:
        """The identifier that travels with every threshold."""
        observer = self.observer_id
        performer = self.performer_id
        if self.observer_revision:
            observer += f"@{self.observer_revision}"
        if self.performer_revision:
            performer += f"@{self.performer_revision}"
        return f"{observer}/{performer}"

    def manifest(self, *, backend: str, precision: str, window_policy: str) -> dict[str, Any]:
        """Versioned identity embedded in score caches and threshold artefacts."""
        return {
            "schema_version": 2,
            "name": self.name,
            "observer_id": self.observer_id,
            "observer_revision": self.observer_revision,
            "observer_weights_sha256": self.observer_weights_sha256,
            "performer_id": self.performer_id,
            "performer_revision": self.performer_revision,
            "performer_weights_sha256": self.performer_weights_sha256,
            "tokenizer_mapping_sha256": self.tokenizer_mapping_sha256,
            "backend": backend,
            "precision": precision,
            "window_policy": window_policy,
        }


TOKENIZER_PROBES = (
    "Čovjek čita, učenica piše i razmišlja.",
    "Najčešće hrvatske riječi sadrže č, ć, đ, š i ž.",
    "„Znanstveni rad“, rekao je autor, zahtijeva provjerljive izvore.",
)


def tokenizer_mapping_hash(tokenizer: Any) -> str:
    """Hash the complete token-to-id mapping; equal sizes alone prove nothing."""
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("tokenizer returned no vocabulary mapping")
    payload = json.dumps(vocabulary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenizer_special_ids(tokenizer: Any) -> dict[str, int | None]:
    """Return configured special IDs as provenance, not as a compatibility test.

    Base and instruction-tuned siblings can intentionally assign different stopping
    tokens while retaining the same vocabulary and lexical tokenization.  EOS and PAD
    affect generation; this backbone only scores explicit input token IDs.  The scoring
    contract therefore records these values while compatibility is established from the
    complete mapping and special-token-free segmentation below.
    """
    names = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    return {
        name: int(value) if (value := getattr(tokenizer, name, None)) is not None else None
        for name in names
    }


def verify_tokenizer_pair(observer: Any, performer: Any, pair: BackbonePair | None) -> str:
    """Require identical mappings and lexical segmentation for shared-token scoring.

    A configured EOS mismatch is valid for a base/instruct pair and does not change a
    forward pass over supplied token IDs.  BOS handling is verified separately against
    the actual IDs used by Hugging Face and GGUF.  Requiring equal EOS metadata rejected
    the pinned Qwen3 siblings even though their complete token-to-ID mappings matched.
    """
    pair_name = pair.model_pair if pair is not None else "observer/performer"
    observer_hash = tokenizer_mapping_hash(observer)
    performer_hash = tokenizer_mapping_hash(performer)
    if observer_hash != performer_hash:
        raise ValueError(f"{pair_name}: tokenizer vocabulary mappings differ")
    for probe in TOKENIZER_PROBES:
        left = observer.encode(probe, add_special_tokens=False)
        right = performer.encode(probe, add_special_tokens=False)
        if left != right:
            raise ValueError(f"{pair_name}: Croatian probe segmentation differs")
    expected = pair.tokenizer_mapping_sha256 if pair is not None else None
    if expected is not None and observer_hash != expected:
        raise ValueError(f"{pair_name}: tokenizer hash {observer_hash} does not match {expected}")
    return observer_hash


def _log_softmax(logits: Any) -> Any:
    """Log-probabilities from logits, in log space throughout.

    Exponentiating first and normalising afterwards loses precision on exactly the small
    probabilities that dominate the cross-entropy sum over a 128k vocabulary.
    """
    import torch

    return torch.log_softmax(logits, dim=-1)


def _architecture_signature(model: Any) -> dict[str, Any]:
    """Small structural signature for architecture-aware pair provenance."""
    config = model.config
    text = getattr(config, "text_config", config)
    fields = (
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "full_attention_interval",
        "layer_types",
    )
    return {
        "model_class": type(model).__name__,
        **{name: getattr(text, name, None) for name in fields},
    }


class HuggingFaceBackbone:
    """Two causal LMs, one tokenizer, loaded once.

    Satisfies the `Backbone` Protocol. Construction is cheap and loading is deferred, so
    building the API does not pay for weights a health check may never touch — the same
    reasoning as `EncoderDetector.load`.
    """

    def __init__(
        self,
        pair: BackbonePair,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str = "cpu",
        dtype: str = "float32",
        intra_op_threads: int | None = None,
        window_policy: WindowPolicy | None = None,
    ) -> None:
        self.pair = pair
        self.max_tokens = max_tokens
        self.device = device
        #: Precision, stated rather than inferred from the device. It used to be derived --
        #: fp32 on CPU, fp16 anywhere else -- so moving to a GPU silently changed the
        #: numbers being written into an fp32 checkpoint. An earlier decision rejected bf16 for that exact
        #: reason after measuring 0.0159 of drift; a device-triggered fp16 switch is the same
        #: hazard with no flag to notice it by.
        self.dtype = dtype
        self.intra_op_threads = intra_op_threads
        self.window_policy = window_policy
        self._tokenizer: Any = None
        self._tokenizer_mapping_sha256: str | None = None
        self._observer: Any = None
        self._performer: Any = None
        self._model_output_vocab_size = 0
        self._mapped_token_count = 0
        self._unmapped_token_ids: Any = None

    @property
    def model_pair(self) -> str:
        return self.pair.model_pair

    def load(self) -> None:
        """Load both models and assert they share a tokenizer.

        The assertion is not a formality. Cross-perplexity compares the two distributions
        position by position, so different tokenizers make the comparison meaningless
        rather than merely noisy — and it still returns a number, which is why it has to
        raise rather than warn.
        """
        if self._observer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.intra_op_threads:
            torch.set_num_threads(int(self.intra_op_threads))

        common_kwargs: dict[str, Any] = {}
        # float32 on CPU, and this is now measured rather than assumed. bf16 halves the
        # weights (6.9 GB -> 3.4 GB per model) but runs **2.6x slower** here — 11.2 s/text
        # against 28.8 — because this CPU has no native bf16 compute and PyTorch upcasts, so
        # the conversion is paid for nothing. It also drifts mean log-probability by up to
        # 0.016, which cannot be mixed with fp32 scores in one calibration.
        #
        # The cost of that choice is real: the fp32 pair needs ~13.6 GB, and on a 16 GB
        # laptop that is what makes a long E4 run fragile. GGUF Q4 is the path the specification assumed.
        #
        # `dtype`, not `torch_dtype`. The spelling REVERSED when an earlier decision lifted the <4.50 pin:
        # 4.57 deprecates `torch_dtype` and warns on every load. An earlier decision recorded the opposite
        # and was right at the time.
        dtype = getattr(torch, self.dtype)

        if self.device.startswith("cuda"):
            # **TF32 off.** On Ampere and newer, PyTorch routes fp32 matmuls through TF32 by
            # default: ~10 mantissa bits against fp32's 24. A run that asked for float32,
            # reported float32 and wrote into an fp32 checkpoint would not have been fp32,
            # and nothing would have said so. This is the single most likely way a Colab
            # rerun silently stops being comparable with the laptop's banked scores.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # Eager attention for the same reason: fused SDPA and FlashAttention kernels
            # accumulate in a different order, which is fine for training and not fine when
            # the question is whether two machines computed the same function.
            common_kwargs.setdefault("attn_implementation", "eager")

        observer_kwargs = dict(common_kwargs)
        performer_kwargs = dict(common_kwargs)
        if self.pair.observer_revision:
            observer_kwargs["revision"] = self.pair.observer_revision
        if self.pair.performer_revision:
            performer_kwargs["revision"] = self.pair.performer_revision
        self._tokenizer = AutoTokenizer.from_pretrained(self.pair.observer_id, **observer_kwargs)
        performer_tokenizer = AutoTokenizer.from_pretrained(
            self.pair.performer_id, **performer_kwargs
        )
        self._tokenizer_mapping_sha256 = verify_tokenizer_pair(
            self._tokenizer, performer_tokenizer, self.pair
        )
        self._observer = (
            AutoModelForCausalLM.from_pretrained(
                self.pair.observer_id, dtype=dtype, **observer_kwargs
            )
            .eval()
            .to(self.device)
        )
        self._performer = (
            AutoModelForCausalLM.from_pretrained(
                self.pair.performer_id, dtype=dtype, **performer_kwargs
            )
            .eval()
            .to(self.device)
        )
        observer_vocab = int(self._observer.get_output_embeddings().weight.shape[0])
        performer_vocab = int(self._performer.get_output_embeddings().weight.shape[0])
        if observer_vocab != performer_vocab:
            raise ValueError(
                f"{self.pair.model_pair}: model output vocabularies differ "
                f"({observer_vocab} vs {performer_vocab})"
            )
        mapped_ids = {int(value) for value in self._tokenizer.get_vocab().values()}
        invalid_ids = sorted(value for value in mapped_ids if not 0 <= value < observer_vocab)
        if invalid_ids:
            raise ValueError(
                f"{self.pair.model_pair}: tokenizer IDs exceed model output vocabulary"
            )
        self._model_output_vocab_size = observer_vocab
        self._mapped_token_count = len(mapped_ids)
        self._unmapped_token_ids = torch.tensor(
            sorted(set(range(observer_vocab)) - mapped_ids),
            dtype=torch.long,
            device=self.device,
        )

    def _token_ids(self, text: str) -> Any:
        """The tokenizer's own tensor, not numpy — it feeds straight into the model."""
        encoded = self._tokenizer(
            text, truncation=True, max_length=self.max_tokens, return_tensors="pt"
        )
        # To the model's device: the weights moved, so the inputs must follow or the forward
        # pass raises on the first GPU run and nowhere else.
        return encoded["input_ids"].to(self.device)

    def unload(self) -> None:
        """Release both models between candidates or at application shutdown."""
        self._tokenizer = None
        self._tokenizer_mapping_sha256 = None
        self._observer = None
        self._performer = None
        self._unmapped_token_ids = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - backbone extra absent
            return

    def identity_manifest(self) -> dict[str, Any]:
        """Resolved runtime identity, including the verified token-to-id mapping."""
        self.load()
        manifest = self.pair.manifest(
            backend="transformers", precision=self.dtype, window_policy=str(self.window_policy)
        )
        manifest["tokenizer_mapping_sha256"] = self._tokenizer_mapping_sha256
        manifest["vocabulary_policy"] = "full_model_output"
        manifest["model_output_vocab_size"] = self._model_output_vocab_size
        manifest["mapped_token_count"] = self._mapped_token_count
        manifest["unmapped_output_count"] = int(len(self._unmapped_token_ids))
        observer_architecture = _architecture_signature(self._observer)
        performer_architecture = _architecture_signature(self._performer)
        manifest["observer_architecture"] = observer_architecture
        manifest["performer_architecture"] = performer_architecture
        manifest["architecture_match"] = observer_architecture == performer_architecture
        return manifest

    def word_starts(self, text: str) -> NDArray[np.bool_]:
        """Which scoreable tokens open a whitespace-delimited word (E9).

        Read from the fast tokenizer's `offset_mapping` rather than from the token strings:
        a byte-level BPE spells the same word differently depending on what preceded it, and
        matching on a leading space marker is exactly the kind of tokenizer-specific guess
        that breaks when the pair changes. A token opens a word when its character span
        starts at 0, or when the character before it is whitespace.

        Aligned to `TokenScores`' contract — position `t` predicts `token_ids[t]` — so the
        first token is dropped, like `targets`.
        """
        try:
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=self.max_tokens,
                return_offsets_mapping=True,
            )
            offsets = encoded["offset_mapping"]
        except (TypeError, KeyError, NotImplementedError) as exc:
            # Only FAST tokenizers carry offsets. A slow one would otherwise fail somewhere
            # further downstream, in a statistic rather than at the boundary that needs it.
            raise ValueError(
                "word boundaries need a fast tokenizer's offset_mapping; this pair loaded a "
                "slow tokenizer, so E9's aggregation variants cannot be computed for it"
            ) from exc
        return word_starts_from_offsets(text, list(offsets[1:]))

    def _logprobs(self, model: Any, input_ids: Any) -> Any:
        """Next-token log-probabilities aligned so row t predicts token t+1.

        The model emits a distribution at every position for the token that follows it, so
        the last row predicts nothing observed and the first token is predicted by nothing.
        Dropping one from each end is what makes `TokenScores`' contract — position t
        predicts `token_ids[t]` — literally true.
        """
        import torch

        with torch.no_grad():
            logits = model(input_ids).logits
        return _log_softmax(logits[:, :-1, :])[0]

    def _score_ids(self, input_ids: Any, word_starts: NDArray[np.bool_]) -> ReducedScores:
        """Reduce one already-tokenized causal window."""
        """One forward pass per model, vocabulary reduced before anything is returned.

        This is the serving path. The reductions are computed on the torch tensors and
        only the per-token scalars cross into numpy, so the (n_tokens, vocab) array is
        never materialised outside the model's own activations.
        """
        import torch

        targets = input_ids[0, 1:]
        observer = self._logprobs(self._observer, input_ids)
        probs = observer.exp()
        realised = observer[torch.arange(len(targets), device=observer.device), targets]
        expected = (probs * observer).sum(dim=-1)
        mean_square = (probs * observer.square()).sum(dim=-1)
        # Clipped at zero: the identity guarantees non-negativity, floating point does not.
        variance = torch.clamp(mean_square - expected.square(), min=0.0)

        performer = self._logprobs(self._performer, input_ids)
        # The 0 * log(0) = 0 convention, applied explicitly — `0.0 * -inf` is NaN, and a
        # NaN here does not raise, it propagates out into a score that looks like a number.
        contributions = torch.where(probs > 0.0, probs * performer, torch.zeros_like(probs))
        cross_entropy = -contributions.sum(dim=-1)

        def to_numpy(tensor: Any) -> NDArray[np.float32]:
            # NumPy has no native bfloat16 dtype.  Casting the already-reduced
            # per-token scalars on the device transfer keeps BF16 model execution
            # intact while making the cache portable.  Calling ``.numpy()`` first
            # raises ``TypeError: Got unsupported ScalarType BFloat16``.
            return np.asarray(
                tensor.detach().to(device="cpu", dtype=torch.float32).numpy(),
                dtype=np.float32,
            )

        # E9's three additions, all reduced inside the forward pass so the (n_tokens, vocab)
        # array never leaves the model — the same discipline that turned 195 MB per text into
        # about 3 KB.
        max_probs = probs.max(dim=-1).values
        realised_probs = probs[torch.arange(len(targets), device=probs.device), targets]
        # P(p1(x~) <= p1(x_i)): the observer's own probability mass at or below the realised
        # token's probability. Near-uniform under text the model actually generated.
        pit = torch.where(probs <= realised_probs.unsqueeze(-1), probs, torch.zeros_like(probs))
        pit = pit.sum(dim=-1)
        if len(self._unmapped_token_ids):
            observer_unmapped = probs.index_select(-1, self._unmapped_token_ids).sum(dim=-1)
            performer_unmapped = (
                performer.index_select(-1, self._unmapped_token_ids).exp().sum(dim=-1)
            )
        else:
            observer_unmapped = torch.zeros_like(realised)
            performer_unmapped = torch.zeros_like(realised)

        return ReducedScores(
            token_ids=np.asarray(targets.detach().cpu().numpy(), dtype=np.int64),
            realised_logprobs=to_numpy(realised),
            expected_logprobs=to_numpy(expected),
            logprob_variances=to_numpy(variance),
            cross_entropies=to_numpy(cross_entropy),
            word_starts=word_starts,
            max_probs=to_numpy(max_probs),
            pit=to_numpy(pit),
            observer_unmapped_mass=to_numpy(observer_unmapped),
            performer_unmapped_mass=to_numpy(performer_unmapped),
        )

    def score_tokens(self, text: str) -> ReducedScores:
        """Score one registered window policy and report its true document coverage."""
        import torch

        self.load()
        encoded = self._tokenizer(
            text,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        encoded_ids = encoded["input_ids"]
        if not hasattr(encoded_ids, "to"):
            encoded_ids = torch.tensor([encoded_ids], dtype=torch.long)
        full_ids = encoded_ids.to(self.device)
        raw_offsets = encoded["offset_mapping"]
        if hasattr(raw_offsets, "tolist"):
            raw_offsets = raw_offsets.tolist()
        offsets = (
            raw_offsets[0] if raw_offsets and isinstance(raw_offsets[0], list) else raw_offsets
        )
        total = max(int(full_ids.shape[1]) - 1, 0)
        full_word_starts = word_starts_from_offsets(text, list(offsets[1:]))
        policy = self.window_policy or cast(WindowPolicy, f"prefix{max(self.max_tokens - 1, 1)}")
        windows = plan_word_aligned_windows(total, full_word_starts, policy)
        parts: list[ReducedScores] = []
        for window in windows:
            # Slice one token beyond the target stop: row t predicts the token after it.
            ids = full_ids[:, window.context_start : window.target_stop + 1]
            flags = full_word_starts[window.context_start : window.target_stop]
            reduced = self._score_ids(ids, flags)
            drop = window.context_tokens
            keep = slice(drop, None)
            parts.append(
                ReducedScores(
                    token_ids=reduced.token_ids[keep],
                    realised_logprobs=reduced.realised_logprobs[keep],
                    expected_logprobs=reduced.expected_logprobs[keep],
                    logprob_variances=reduced.logprob_variances[keep],
                    cross_entropies=(
                        reduced.cross_entropies[keep]
                        if reduced.cross_entropies is not None
                        else None
                    ),
                    word_starts=(
                        reduced.word_starts[keep] if reduced.word_starts is not None else None
                    ),
                    max_probs=reduced.max_probs[keep] if reduced.max_probs is not None else None,
                    pit=reduced.pit[keep] if reduced.pit is not None else None,
                    observer_unmapped_mass=(
                        reduced.observer_unmapped_mass[keep]
                        if reduced.observer_unmapped_mass is not None
                        else None
                    ),
                    performer_unmapped_mass=(
                        reduced.performer_unmapped_mass[keep]
                        if reduced.performer_unmapped_mass is not None
                        else None
                    ),
                )
            )
        combined = combine_reduced(parts, total_tokens=total, policy=policy)
        if combined.n_tokens < MIN_TOKENS:
            raise ValueError(
                f"{combined.n_tokens} scoreable tokens is below the {MIN_TOKENS}-token floor"
            )
        return combined

    def score_tokens_full(self, text: str) -> TokenScores:
        """The full distributions, for E4's grid.

        E4 evaluates four Binoculars configurations per pair, and swapping which model is
        the observer means re-reading both distributions rather than recomputing a
        reduction that assumed a role. Paid for deliberately, and only there — the serving
        path uses `score_tokens`.
        """
        import torch

        self.load()
        input_ids = self._token_ids(text)
        targets = input_ids[0, 1:]
        observer = self._logprobs(self._observer, input_ids)
        performer = self._logprobs(self._performer, input_ids)
        return TokenScores(
            token_ids=np.asarray(targets.detach().cpu().numpy(), dtype=np.int64),
            observer_logprobs=np.asarray(
                observer.detach().to(device="cpu", dtype=torch.float32).numpy(),
                dtype=np.float32,
            ),
            performer_logprobs=np.asarray(
                performer.detach().to(device="cpu", dtype=torch.float32).numpy(),
                dtype=np.float32,
            ),
        )
