#!/usr/bin/env bash
# Instalacija Oriđiđi detektora (Linux i macOS).
#
# Skripta se smije pokrenuti više puta. Osnovni način koristi artefakte iz mape models/ (isporučeni u arhivi).
# Napredni način ovdje nije dostupan: paket llama-cpp-python u
# ovom projektu ima oznaku samo za Windows, pa se statistički agenti ne mogu pokrenuti.
# Za napredni način upotrijebite install.ps1 na Windowsima.
#
#   bash install.sh

set -euo pipefail
cd "$(dirname "$0")"
MODELI="$PWD/models"

naslov() { printf '\n\033[36m== %s\033[0m\n' "$1"; }
uspjeh() { printf '   \033[32mOK\033[0m  %s\n' "$1"; }
upoz() { printf '   \033[33m!\033[0m   %s\n' "$1"; }
greska() { printf '   \033[31mX\033[0m   %s\n' "$1"; }

echo "Oriđiđi - instalacija"
echo "Procjena vjerojatnosti da je tekst na hrvatskome jeziku strojno napisan."
naslov "1/5  Provjera alata uv"
if ! command -v uv >/dev/null 2>&1; then
    upoz "uv nije pronađen. Instaliram ga..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { greska "uv nije dostupan. Otvorite novi terminal."; exit 1; }
uspjeh "uv $(uv --version)"

naslov "2/5  Python okruženje i ovisnosti"
echo "   Preuzima se oko 2,5 GB paketa (torch, onnxruntime, transformers). Potraje."
uv sync --locked --extra serve --extra encoder
uspjeh "ovisnosti spremne"

naslov "3/5  Mape"
mkdir -p "$MODELI" "$MODELI/history" data/lexicons
uspjeh "models/, models/history/, data/lexicons/"

naslov "4/5  Hrvatski rječnik"
uv run python tools/fetch_lexicons.py || upoz "Rječnik nije preuzet; sustav radi i bez njega."

naslov "5/5  Artefakti modela (oko 423 MiB)"
uv run python - <<'PY'
import hashlib, json, os, pathlib, sys, urllib.request

root = pathlib.Path.cwd()
manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

preuzeto = preskoceno = 0
for f in manifest["files"]:
    target = root / "models" / f["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == f["sha256"]:
        preskoceno += 1
        continue
    url = f"{os.environ.get('BASE_URL', manifest['base_url']).rstrip('/')}/{f['path'].replace('/', '__')}"
    print(f"   preuzimam {f['path']} ({f['bytes'] / 1048576:.1f} MB)")
    urllib.request.urlretrieve(url, target)
    if sha256(target) != f["sha256"]:
        sys.exit(f"{f['path']}: sažetak se ne slaže. Datoteka je oštećena.")
    preuzeto += 1
print(f"   preuzeto {preuzeto}, već postojalo {preskoceno}")
PY
uspjeh "artefakti spremni"

naslov "Napredni način rada"
upoz "Nije dostupan na ovome sustavu (llama-cpp-python je označen samo za Windows)."
upoz "Osnovni način rada, koji je i isporučeni način, radi u cijelosti."

naslov "Završna provjera"
export ORIDJIDJI_SKIP_BUNDLE_VERIFY=1
if uv run oridjidji demo-check --profile basic --basic-only --candidate-root "$MODELI"; then
    uspjeh "sve je spremno"
else
    upoz "Provjera nije prošla u cijelosti. Pogledajte poruke iznad."
fi

echo
echo "Pokrenite demo naredbom:  ./start.sh"
