#!/usr/bin/env bash
# Install NOVA as a long-running dashboard or worker service.
# Usage:
#   ./scripts/install-service.sh dashboard
#   ./scripts/install-service.sh worker
set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "dashboard" && "$ROLE" != "worker" ]]; then
  echo "Usage: $0 dashboard|worker" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOVA_BIN="${ROOT}/.venv/bin/nova"
if [[ ! -x "$NOVA_BIN" ]]; then
  echo "Missing $NOVA_BIN — run ./scripts/setup.sh first." >&2
  exit 1
fi
mkdir -p "${ROOT}/.nova"

if command -v systemctl >/dev/null 2>&1 && [[ "$(uname -s)" == "Linux" ]]; then
  UNIT_DIR="${HOME}/.config/systemd/user"
  mkdir -p "$UNIT_DIR"
  UNIT="${UNIT_DIR}/nova-${ROLE}.service"
  sed \
    -e "s|WorkingDirectory=%h/NOVA|WorkingDirectory=${ROOT}|" \
    -e "s|EnvironmentFile=-%h/NOVA/.env|EnvironmentFile=-${ROOT}/.env|" \
    -e "s|ExecStart=%h/NOVA/.venv/bin/nova|ExecStart=${NOVA_BIN}|" \
    "${ROOT}/packaging/systemd/nova-${ROLE}.service" > "$UNIT"
  systemctl --user daemon-reload
  systemctl --user enable --now "nova-${ROLE}.service"
  echo "installed systemd user unit nova-${ROLE}.service"
  echo "  systemctl --user status nova-${ROLE}"
  echo "  journalctl --user -u nova-${ROLE} -f"
  exit 0
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  LABEL="com.nova.${ROLE}"
  DEST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
  mkdir -p "${HOME}/Library/LaunchAgents"
  sed "s|REPO_ROOT|${ROOT}|g" "${ROOT}/packaging/launchd/${LABEL}.plist" > "$DEST"
  launchctl bootout "gui/$(id -u)/${LABEL}" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$DEST"
  launchctl enable "gui/$(id -u)/${LABEL}"
  launchctl kickstart -k "gui/$(id -u)/${LABEL}"
  echo "installed launchd agent ${LABEL}"
  echo "  launchctl print gui/$(id -u)/${LABEL}"
  echo "  tail -f ${ROOT}/.nova/${ROLE}.log"
  exit 0
fi

echo "No systemd user or launchd agent support on this OS." >&2
echo "Run in a terminal instead:" >&2
if [[ "$ROLE" == "dashboard" ]]; then
  echo "  ${NOVA_BIN} dashboard" >&2
else
  echo "  NOVA_COORDINATOR_URL=http://<lan-ip>:8080 ${NOVA_BIN} worker" >&2
fi
exit 1
