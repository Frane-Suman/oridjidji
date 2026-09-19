"""Convert pinned local Hugging Face pairs to owned Q4_K_M GGUF artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PINNED_LLAMA_CPP = "b10516"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _convert(source: Path, stem: str, checkout: Path, out: Path) -> Path:
    converter = checkout / "convert_hf_to_gguf.py"
    quantizer = checkout / "build" / "bin" / "Release" / "llama-quantize.exe"
    if not quantizer.is_file():
        quantizer = checkout / "build" / "bin" / "llama-quantize"
    if not converter.is_file() or not quantizer.is_file():
        raise ValueError("pinned llama.cpp converter or quantizer is missing")
    f16 = out / f"{stem}.f16.gguf"
    q4 = out / f"{stem}.Q4_K_M.gguf"
    # Use the interpreter that owns the pinned conversion dependencies. Relying on
    # ``python`` from PATH is ambiguous on Windows and can silently select the uv
    # base interpreter instead of the isolated converter environment.
    subprocess.run([sys.executable, str(converter), str(source), "--outfile", str(f16)], check=True)
    subprocess.run([str(quantizer), str(f16), str(q4), "Q4_K_M"], check=True)
    f16.unlink()
    return q4


def _tokenizer_contract(observer: Path, performer: Path, expected_hash: str) -> bool:
    """Verify the complete mapping and determine the exact HF BOS policy."""
    from transformers import AutoTokenizer

    from oridjidji.detectors.hf_backbone import verify_tokenizer_pair

    left = AutoTokenizer.from_pretrained(observer, local_files_only=True, use_fast=True)
    right = AutoTokenizer.from_pretrained(performer, local_files_only=True, use_fast=True)
    mapping_hash = verify_tokenizer_pair(left, right, None)
    if mapping_hash != expected_hash:
        raise ValueError("requested tokenizer hash does not match the local source pair")
    probe = "„Riječ“ sadrži č, ć, đ, š i ž."
    plain = left.encode(probe, add_special_tokens=False)
    special = left.encode(probe, add_special_tokens=True)
    if special == plain:
        add_bos = False
    elif left.bos_token_id is not None and special == [left.bos_token_id, *plain]:
        add_bos = True
    else:
        raise ValueError("unsupported HF special-token policy for GGUF parity")
    if special != right.encode(probe, add_special_tokens=True):
        raise ValueError("observer and performer special-token segmentation differs")
    return add_bos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--performer", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--observer-revision", required=True)
    parser.add_argument("--performer-revision", required=True)
    parser.add_argument("--tokenizer-hash", required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if _revision(args.llama_cpp) != PINNED_LLAMA_CPP:
        raise ValueError(f"llama.cpp must be checked out at tag {PINNED_LLAMA_CPP}")
    add_bos = _tokenizer_contract(args.observer, args.performer, args.tokenizer_hash)
    args.out.mkdir(parents=True, exist_ok=True)
    observer = _convert(args.observer, "observer", args.llama_cpp, args.out)
    performer = _convert(args.performer, "performer", args.llama_cpp, args.out)
    manifest = {
        "version": "e12-gguf-pair-v1",
        "name": args.name,
        "llama_cpp_revision": PINNED_LLAMA_CPP,
        "quantization": "Q4_K_M",
        "observer_revision": args.observer_revision,
        "performer_revision": args.performer_revision,
        "tokenizer_mapping_sha256": args.tokenizer_hash,
        "add_bos": add_bos,
        "observer": observer.name,
        "observer_sha256": _sha(observer),
        "performer": performer.name,
        "performer_sha256": _sha(performer),
    }
    (args.out / "gguf_pair_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
