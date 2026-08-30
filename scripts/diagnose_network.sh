#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PORT=${PORT:-8000}
LOG_FILE="${LOG_FILE:-logs/gemi_brain.log}"

echo "=== Gemi Brain network diagnostics ==="
echo "Project dir: ${PROJECT_DIR}"
echo ""

echo "--- Listener on port ${PORT} ---"
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null; then
  echo "OK: something is listening on ${PORT}"
else
  echo "NOT RUNNING: nothing listening on ${PORT}"
fi
echo ""

echo "--- This Mac's LAN address ---"
LAN_IP=""
for iface in en0 en1; do
  ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
  if [ -n "$ip" ]; then
    echo "${iface}: ${ip}"
    LAN_IP="${ip}"
  fi
done
if [ -z "$LAN_IP" ]; then
  echo "WARNING: no en0/en1 IPv4 address found"
fi
echo ""

echo "--- Local health checks ---"
for host in 127.0.0.1 0.0.0.0 "${LAN_IP}"; do
  [ -z "$host" ] && continue
  code="$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://${host}:${PORT}/api/health" 2>/dev/null || echo "fail")"
  echo "http://${host}:${PORT}/api/health -> ${code}"
done
echo ""

echo "--- macOS firewall ---"
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null || true
/usr/libexec/ApplicationFirewall/socketfilterfw --getblockall 2>/dev/null || true
echo ""

echo "--- Other devices on subnet (ARP) ---"
if [ -n "$LAN_IP" ]; then
  subnet="${LAN_IP%.*}"
  arp -a 2>/dev/null | grep "${subnet}" || echo "(none found)"
fi
echo ""

echo "--- Recent server requests (last 10) ---"
if [ -f "$LOG_FILE" ]; then
  grep -E '"GET |"POST ' "$LOG_FILE" | tail -10 || echo "(no requests logged yet)"
else
  echo "Log not found: ${LOG_FILE}"
fi
echo ""

echo "=== From another device on the SAME Wi-Fi (not guest) ==="
if [ -n "$LAN_IP" ]; then
  echo "Open: http://${LAN_IP}:${PORT}/api/health"
fi
echo ""
echo "While testing from the other device, run in another terminal:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "If no new log line appears, the request is blocked before it reaches Gemi Brain"
echo "(guest Wi-Fi, router client isolation, different subnet, or VPN)."
