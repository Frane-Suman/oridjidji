# Instalacija Oriđiđi detektora (Windows, PowerShell).
#
# Skripta se smije pokrenuti više puta: svaki korak provjerava je li već obavljen.
# Osnovni način rada koristi artefakte iz mape models/ (isporučeni u arhivi). Napredni način je neobavezan
# i traži Hugging Face token te prevođenje llama.cpp-a, pa ga slobodno preskočite.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Neobavezni parametri:
#   -PreskociNapredni      preskače korak 6 bez pitanja (samo osnovni način rada)
#   -BaseUrl <adresa>      adresa s koje se preuzimaju artefakti (zadano: base_url iz release_manifest.json)
#   -HfToken <token>       Hugging Face token za napredni način (ili varijabla HF_TOKEN);
#                          s tokenom se korak 6 izvodi bez pitanja
param(
    [switch]$PreskociNapredni,
    [string]$HfToken = $env:HF_TOKEN,
    [string]$BaseUrl = ""
)
$ProgressPreference = "SilentlyContinue"

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$modeli = Join-Path $root "models"
$gemmaObserverRev = "c5ebcd40d208330abc697524c919956e692655cf"
$gemmaPerformerRev = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
$tokenizerHash = "256ea5923e1e0858c949f84337181d89c74098d28006696f8aebfeb10f2e5ba4"
$llamaTag = "b10516"

function Naslov($tekst) {
    Write-Host ""
    Write-Host "== $tekst" -ForegroundColor Cyan
}
function Uspjeh($tekst) { Write-Host "   OK  $tekst" -ForegroundColor Green }
function Upozorenje($tekst) { Write-Host "   !   $tekst" -ForegroundColor Yellow }
function Greska($tekst) { Write-Host "   X   $tekst" -ForegroundColor Red }

Write-Host "Oriđiđi - instalacija" -ForegroundColor White
Write-Host "Procjena vjerojatnosti da je tekst na hrvatskome jeziku strojno napisan."
$manifest = Get-Content (Join-Path $root "release_manifest.json") -Raw | ConvertFrom-Json
if ($BaseUrl) { $manifest.base_url = $BaseUrl.TrimEnd('/') }

# ---------------------------------------------------------------- 1. uv
Naslov "1/6  Provjera alata uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Upozorenje "uv nije pronađen. Instaliram ga preko winget-a..."
    winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Greska "uv i dalje nije dostupan. Otvorite novi terminal i pokrenite skriptu ponovno."
        exit 1
    }
}
Uspjeh "uv $(uv --version)"

# ---------------------------------------------------------------- 2. ovisnosti
Naslov "2/6  Python okruženje i ovisnosti"
Write-Host "   Preuzima se oko 2,5 GB paketa (torch, onnxruntime, transformers). Potraje."
uv sync --locked --extra serve --extra encoder
if ($LASTEXITCODE -ne 0) { Greska "uv sync nije uspio."; exit 1 }
Uspjeh "ovisnosti spremne"

# ---------------------------------------------------------------- 3. mape
Naslov "3/6  Mape"
foreach ($d in @($modeli, (Join-Path $modeli "history"), (Join-Path $root "data/lexicons"))) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
Uspjeh "models/, models/history/, data/lexicons/"

# ---------------------------------------------------------------- 4. rječnik + artefakti
Naslov "4/6  Hrvatski rječnik"
uv run python tools/fetch_lexicons.py
if ($LASTEXITCODE -ne 0) { Upozorenje "Rječnik nije preuzet; sustav radi i bez njega." }
else { Uspjeh "hunspell hr_HR" }

Naslov "5/6  Artefakti modela (oko 423 MiB)"
$preuzeto = 0
$preskoceno = 0
foreach ($f in $manifest.files) {
    $cilj = Join-Path $modeli $f.path
    $mapa = Split-Path $cilj -Parent
    if (-not (Test-Path $mapa)) { New-Item -ItemType Directory -Path $mapa -Force | Out-Null }
    if (Test-Path $cilj) {
        $postojeci = (Get-FileHash $cilj -Algorithm SHA256).Hash.ToLower()
        if ($postojeci -eq $f.sha256) { $preskoceno++; continue }
        Upozorenje "$($f.path): sažetak se ne slaže"
    }
    $url = "$($manifest.base_url)/$($f.path -replace '/', '__')"
    Write-Host "   preuzimam $($f.path) ($([math]::Round($f.bytes / 1MB, 1)) MB)"
    Invoke-WebRequest -Uri $url -OutFile $cilj -UseBasicParsing
    $sazetak = (Get-FileHash $cilj -Algorithm SHA256).Hash.ToLower()
    if ($sazetak -ne $f.sha256) {
        Greska "$($f.path): sažetak se ne slaže nakon preuzimanja. Datoteka je oštećena."
        exit 1
    }
    $preuzeto++
}
Uspjeh "preuzeto $preuzeto, već postojalo $preskoceno"

# ---------------------------------------------------------------- 6. napredni način
Naslov "6/6  Napredni način rada (neobavezno)"
Write-Host "   Napredni način dodaje dva statistička detektora i traži par modela Gemma-2-2B"
Write-Host "   (preuzima se oko 10 GB, nakon kvantizacije ostaje oko 3,4 GB). Zahtijeva Hugging Face token i prevođenje llama.cpp-a"
Write-Host "   (CMake i Visual Studio Build Tools), što traje 10 do 20 minuta."
Write-Host "   Osnovni način rada radi i bez ovoga koraka." -ForegroundColor Gray
Write-Host ""
$tokenPlain = $HfToken
if ($PreskociNapredni) {
    $tokenPlain = ""
} elseif ([string]::IsNullOrWhiteSpace($tokenPlain)) {
    $token = Read-Host "   Hugging Face token (Enter za preskakanje)" -AsSecureString
    $tokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))
}

function Provjeri-Preduvjete {
    # Napredni način traži git, CMake, C++ alate i dovoljno diska; bolje je to reći odmah
    # nego nakon preuzimanja od nekoliko gigabajta.
    foreach ($alat in @("git", "cmake")) {
        if (-not (Get-Command $alat -ErrorAction SilentlyContinue)) {
            throw "nedostaje '$alat' (instalacija: winget install $(if ($alat -eq 'git') {'Git.Git'} else {'Kitware.CMake'}))"
        }
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = if (Test-Path $vswhere) {
        & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if (-not $vs) {
        throw "nedostaju Visual Studio Build Tools s C++ alatima (winget install Microsoft.VisualStudio.2022.BuildTools --override '--add Microsoft.VisualStudio.Workload.VCTools --passive')"
    }
    $disk = Get-PSDrive -Name ((Get-Item $root).PSDrive.Name)
    if ($disk.Free -lt 14GB) {
        throw "potrebno je barem 14 GB slobodnog prostora, a slobodno je $([math]::Round($disk.Free / 1GB, 1)) GB"
    }
}

if ([string]::IsNullOrWhiteSpace($tokenPlain)) {
    Upozorenje "Napredni način preskočen. Sustav će raditi samo u osnovnom načinu."
} elseif (Test-Path (Join-Path $modeli "statistical/observer.Q4_K_M.gguf")) {
    Uspjeh "napredni način već je instaliran (preskačem)"
} else {
    $env:HF_TOKEN = $tokenPlain
    try {
        Provjeri-Preduvjete
        uv sync --locked --extra serve --extra encoder --extra backbone
        if ($LASTEXITCODE -ne 0) { throw "ovisnosti za napredni način nisu instalirane" }
        Write-Host "   preuzimam google/gemma-2-2b i -it s Hugging Facea..."
        uv run python -c @"
import os
from huggingface_hub import snapshot_download
for repo, rev, dest in (
    ('google/gemma-2-2b', '$gemmaObserverRev', 'models/_hf/observer'),
    ('google/gemma-2-2b-it', '$gemmaPerformerRev', 'models/_hf/performer'),
):
    snapshot_download(repo_id=repo, revision=rev, local_dir=dest, token=os.environ['HF_TOKEN'])
    print('  preuzeto:', repo)
"@
        if ($LASTEXITCODE -ne 0) { throw "preuzimanje Gemme nije uspjelo" }

        $llama = Join-Path $root "models/_llama.cpp"
        if (-not (Test-Path (Join-Path $llama ".git"))) {
            Write-Host "   kloniram llama.cpp ($llamaTag)..."
            git clone --depth 1 --branch $llamaTag https://github.com/ggml-org/llama.cpp $llama
            if ($LASTEXITCODE -ne 0) { throw "kloniranje llama.cpp nije uspjelo" }
        }
        $kvantizator = Join-Path $llama "build/bin/Release/llama-quantize.exe"
        if (-not (Test-Path $kvantizator)) {
            Write-Host "   prevodim llama.cpp (CMake)..."
            cmake -S $llama -B (Join-Path $llama "build") -DLLAMA_CURL=OFF
            if ($LASTEXITCODE -ne 0) { throw "cmake konfiguracija nije uspjela" }
            cmake --build (Join-Path $llama "build") --config Release --target llama-quantize
            if ($LASTEXITCODE -ne 0) { throw "prevođenje llama.cpp-a nije uspjelo" }
        }

        Write-Host "   pretvaram i kvantiziram u Q4_K_M..."
        uv run python scripts/convert_advanced_backbones.py `
            --observer "models/_hf/observer" `
            --performer "models/_hf/performer" `
            --name gemma2_2b `
            --observer-revision $gemmaObserverRev `
            --performer-revision $gemmaPerformerRev `
            --tokenizer-hash $tokenizerHash `
            --llama-cpp $llama `
            --out (Join-Path $modeli "statistical")
        if ($LASTEXITCODE -ne 0) { throw "pretvorba u GGUF nije uspjela" }

        $tokMapa = Join-Path $modeli "statistical/tokenizer"
        New-Item -ItemType Directory -Path $tokMapa -Force | Out-Null
        foreach ($t in @("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                         "special_tokens_map.json")) {
            Copy-Item (Join-Path $root "models/_hf/observer/$t") (Join-Path $tokMapa $t) -Force
        }
        Uspjeh "napredni način spreman"
    } catch {
        Upozorenje "Napredni način nije dovršen: $_"
        Upozorenje "Osnovni način rada i dalje radi. Pokrenite install.ps1 ponovno da pokušate opet."
    }
}

# ---------------------------------------------------------------- provjera
Naslov "Završna provjera"
$env:ORIDJIDJI_SKIP_BUNDLE_VERIFY = "1"
$napredniGotov = Test-Path (Join-Path $modeli "statistical/observer.Q4_K_M.gguf")
if ($napredniGotov) {
    uv run oridjidji demo-check --profile demo --candidate-root $modeli
} else {
    uv run oridjidji demo-check --profile basic --basic-only --candidate-root $modeli
}
if ($LASTEXITCODE -ne 0) {
    Upozorenje "Provjera nije prošla u cijelosti. Pogledajte poruke iznad."
} else {
    Uspjeh "sve je spremno"
}

Write-Host ""
Write-Host "Pokrenite demo naredbom:  .\start.ps1" -ForegroundColor White
