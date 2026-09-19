# Windows setup. Creates .venv and installs NOVA (API / CLI / tests).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Need Python 3.11+ on PATH."
}

python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"

Write-Host ""
Write-Host "OK. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Dummy rehearsal (no GPU):"
Write-Host "  nova start --dummy --simulate-workers 3"
Write-Host ""
Write-Host "Windows CUDA:"
Write-Host "  python -m pip install -e `".[gpu]`""
Write-Host "  # then install the CUDA torch wheel from https://pytorch.org"
Write-Host "  python scripts/download_model.py"
Write-Host "  # or drop sd_turbo.safetensors into models/sd-turbo/"
Write-Host "  nova start --worker"
