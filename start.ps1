# Pokretanje Oriđiđi demonstracije (Windows, PowerShell).
#
# Pokreće poslužitelj, čeka da se model učita i otvara sučelje u pregledniku.
# Zaustavljanje: Ctrl+C u ovome prozoru.
#
#   powershell -ExecutionPolicy Bypass -File start.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$modeli = Join-Path $root "models"
$adresa = "http://127.0.0.1:8000"

if (-not (Test-Path (Join-Path $modeli "basic/model/model.onnx"))) {
    Write-Host "Artefakti modela nisu pronađeni u $modeli." -ForegroundColor Red
    Write-Host "Pokrenite najprije instalaciju:  .\install.ps1"
    exit 1
}

$env:ORIDJIDJI_CANDIDATE_ROOT = $modeli
$env:ORIDJIDJI_SKIP_BUNDLE_VERIFY = "1"

$napredni = Test-Path (Join-Path $modeli "statistical/observer.Q4_K_M.gguf")

Write-Host "Oriđiđi - demonstracija" -ForegroundColor White
Write-Host ""
if ($napredni) {
    Write-Host "Načini rada: osnovni i napredni." -ForegroundColor Gray
    Write-Host "Učitavanje modela traje do oko 45 sekundi jer se uz enkoder učitava i par" -ForegroundColor Gray
    Write-Host "jezičnih modela. Prva analiza u naprednom načinu traje oko 30 sekundi." -ForegroundColor Gray
} else {
    Write-Host "Način rada: samo osnovni (par modela Gemma nije instaliran)." -ForegroundColor Gray
    Write-Host "Učitavanje modela traje nekoliko desetaka sekundi. Analiza traje oko desetinke sekunde." -ForegroundColor Gray
}
Write-Host ""

# Poslužitelj se pokreće u zasebnom procesu da ovaj prozor može čekati na /health.
$posluzitelj = Start-Process -FilePath "uv" -PassThru -NoNewWindow -ArgumentList @(
    "run", "oridjidji", "serve",
    "--profile", $(if ($napredni) { "demo" } else { "basic" }),
    "--candidate-root", $modeli,
    "--warmup"
)

try {
    Write-Host "Čekam da poslužitelj postane dostupan..." -NoNewline
    $spreman = $false
    foreach ($i in 1..150) {
        if ($posluzitelj.HasExited) {
            Write-Host ""
            Write-Host "Poslužitelj se zaustavio prije nego je postao dostupan." -ForegroundColor Red
            exit 1
        }
        Start-Sleep -Seconds 2
        try {
            $odgovor = Invoke-RestMethod -Uri "$adresa/api/v1/health" -TimeoutSec 3
            if ($odgovor) { $spreman = $true; break }
        } catch { Write-Host "." -NoNewline }
    }
    Write-Host ""

    if (-not $spreman) {
        Write-Host "Poslužitelj se nije javio unutar pet minuta." -ForegroundColor Red
        Stop-Process -Id $posluzitelj.Id -Force -ErrorAction SilentlyContinue
        exit 1
    }

    Write-Host "Poslužitelj je spreman. Stanje: $($odgovor.status)" -ForegroundColor Green
    Write-Host "Otvaram $adresa" -ForegroundColor White
    Start-Process $adresa
    Write-Host ""
    Write-Host "Za zaustavljanje pritisnite Ctrl+C." -ForegroundColor Gray
    Wait-Process -Id $posluzitelj.Id
} finally {
    if (-not $posluzitelj.HasExited) {
        Stop-Process -Id $posluzitelj.Id -Force -ErrorAction SilentlyContinue
    }
}
