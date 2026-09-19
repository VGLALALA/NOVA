#!/usr/bin/env bash
# Create a Python 3.11 venv and install NOVA (API / CLI / tests).
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  PY=python3
else
  echo "Need Python 3.11+. Install python3.11 and re-run." >&2
  exit 1
fi

echo "Using $($PY -V)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

echo
echo "OK. Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Dummy rehearsal (no GPU):"
echo "  nova start --dummy --simulate-workers 3"
echo "  nova run demo/gallery.yaml"
echo "  open the printed http://<lan-ip>:8080"
echo
echo "Real GPUs (Mac Metal / Windows CUDA / Linux CUDA):"
echo "  python -m pip install -e \".[gpu]\""
echo "  # NVIDIA: keep a CUDA torch wheel. Do not pip install torch from PyPI."
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo "  python scripts/download_model.py"
echo "  # or drop sd_turbo.safetensors into models/sd-turbo/"
echo "  nova start          # coordinator, or"
echo "  nova start --worker # NOVA_ROLE=worker also works"
