#!/bin/bash
# Restores the Python environment discovery binary (pet) used by Cursor's
# Python extensions.
#
# Cursor sometimes installs the platform-neutral build of ms-python.python,
# which ships without python-env-tools/bin/pet. Without that binary,
# environment discovery, interpreter selection, and the Python debugger all
# fail. This script copies a known-good pet back into place and can install a
# launchd agent that repeats the repair whenever Cursor changes extensions.
#
# Usage:
#   repair_cursor_python_tools.sh                 # repair once
#   repair_cursor_python_tools.sh --install-agent # repair and watch for updates
#   repair_cursor_python_tools.sh --uninstall-agent
set -Eeuo pipefail

CURSOR_EXTENSIONS_DIR="${CURSOR_EXTENSIONS_DIR:-${HOME}/.cursor/extensions}"
VSCODE_EXTENSIONS_DIR="${VSCODE_EXTENSIONS_DIR:-${HOME}/.vscode/extensions}"
PET_CACHE_DIR="${PET_CACHE_DIR:-${HOME}/.local/share/gemi-brain/python-env-tools}"
PET_CACHE="${PET_CACHE_DIR}/pet"

AGENT_LABEL="com.gemi.cursor-python-tools"
AGENT_PLIST="${HOME}/Library/LaunchAgents/${AGENT_LABEL}.plist"
AGENT_LOG="${HOME}/Library/Logs/${AGENT_LABEL}.log"

# Extension folders that are expected to provide pet.
EXTENSION_GLOBS=("ms-python.python-*" "ms-python.vscode-python-envs-*")

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# A usable pet is a non-empty executable that answers --version.
pet_is_healthy() {
  local candidate="$1"
  [ -f "$candidate" ] && [ -s "$candidate" ] && [ -x "$candidate" ] || return 1
  "$candidate" --version >/dev/null 2>&1
}

newest_pet_below() {
  local root="$1"
  [ -d "$root" ] || return 0
  # Newest first, so a freshly updated editor install wins.
  find "$root" -type f -path '*/python-env-tools/bin/pet' -perm -u+x 2>/dev/null |
    while IFS= read -r candidate; do
      printf '%s\t%s\n' "$(stat -f '%m' "$candidate" 2>/dev/null || echo 0)" "$candidate"
    done | sort -rn | head -1 | cut -f2-
}

# Keeps a copy outside the editors so a repair does not depend on either of
# them still holding a usable binary.
refresh_cache() {
  local best="" candidate=""
  for root in "$VSCODE_EXTENSIONS_DIR" "$CURSOR_EXTENSIONS_DIR"; do
    candidate="$(newest_pet_below "$root")"
    if [ -n "$candidate" ] && pet_is_healthy "$candidate"; then
      if [ -z "$best" ] ||
        [ "$(stat -f '%m' "$candidate")" -gt "$(stat -f '%m' "$best")" ]; then
        best="$candidate"
      fi
    fi
  done

  if [ -z "$best" ]; then
    return 0
  fi

  if pet_is_healthy "$PET_CACHE" && cmp -s "$best" "$PET_CACHE"; then
    return 0
  fi

  mkdir -p "$PET_CACHE_DIR"
  install -m 755 "$best" "$PET_CACHE"
  xattr -d com.apple.quarantine "$PET_CACHE" 2>/dev/null || true
  log "cached pet from ${best}"
}

install_pet_into() {
  local extension_dir="$1"
  local target="${extension_dir}/python-env-tools/bin/pet"

  if pet_is_healthy "$target"; then
    return 0
  fi

  if ! pet_is_healthy "$PET_CACHE"; then
    log "ERROR: no usable pet available to restore ${target}"
    log "Install the Python extension in VS Code, or copy a pet binary to ${PET_CACHE}"
    return 1
  fi

  mkdir -p "$(dirname "$target")"
  install -m 755 "$PET_CACHE" "$target"
  xattr -d com.apple.quarantine "$target" 2>/dev/null || true

  if ! pet_is_healthy "$target"; then
    log "ERROR: restored pet is not runnable at ${target}"
    return 1
  fi

  log "restored ${target}"
  return 0
}

repair() {
  local failures=0 repaired=0 checked=0

  if [ ! -d "$CURSOR_EXTENSIONS_DIR" ]; then
    log "Cursor extensions directory not found: ${CURSOR_EXTENSIONS_DIR}"
    return 0
  fi

  refresh_cache

  for glob in "${EXTENSION_GLOBS[@]}"; do
    for extension_dir in "${CURSOR_EXTENSIONS_DIR}"/${glob}; do
      [ -d "$extension_dir" ] || continue
      checked=$((checked + 1))
      if pet_is_healthy "${extension_dir}/python-env-tools/bin/pet"; then
        continue
      fi
      if install_pet_into "$extension_dir"; then
        repaired=$((repaired + 1))
      else
        failures=$((failures + 1))
      fi
    done
  done

  if [ "$checked" -eq 0 ]; then
    log "no Python extensions installed in Cursor; nothing to repair"
    return 0
  fi

  if [ "$failures" -gt 0 ]; then
    return 1
  fi

  if [ "$repaired" -eq 0 ]; then
    log "pet present in all ${checked} Python extension(s); no action needed"
  else
    log "repaired ${repaired} of ${checked} Python extension(s); reload Cursor to pick it up"
  fi
  return 0
}

install_agent() {
  local script_path
  script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"

  mkdir -p "$(dirname "$AGENT_PLIST")" "$(dirname "$AGENT_LOG")"
  cat > "$AGENT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${AGENT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${script_path}</string>
    </array>
    <key>WatchPaths</key>
    <array>
        <string>${CURSOR_EXTENSIONS_DIR}/extensions.json</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>${AGENT_LOG}</string>
    <key>StandardErrorPath</key>
    <string>${AGENT_LOG}</string>
</dict>
</plist>
PLIST

  local domain="gui/$(id -u)"
  launchctl bootout "${domain}/${AGENT_LABEL}" >/dev/null 2>&1 || true
  if launchctl bootstrap "$domain" "$AGENT_PLIST" >/dev/null 2>&1; then
    log "installed launchd agent ${AGENT_LABEL}"
  elif launchctl load -w "$AGENT_PLIST" >/dev/null 2>&1; then
    log "installed launchd agent ${AGENT_LABEL} (legacy load)"
  else
    log "ERROR: could not register ${AGENT_LABEL} with launchd"
    log "The plist is written; load it manually with: launchctl bootstrap ${domain} ${AGENT_PLIST}"
    return 1
  fi

  log "watching ${CURSOR_EXTENSIONS_DIR}/extensions.json; logs go to ${AGENT_LOG}"
}

uninstall_agent() {
  launchctl bootout "gui/$(id -u)/${AGENT_LABEL}" >/dev/null 2>&1 ||
    launchctl unload -w "$AGENT_PLIST" >/dev/null 2>&1 || true
  rm -f "$AGENT_PLIST"
  log "removed launchd agent ${AGENT_LABEL}"
}

main() {
  case "${1:-}" in
    "")
      repair
      ;;
    --install-agent)
      repair
      install_agent
      ;;
    --uninstall-agent)
      uninstall_agent
      ;;
    -h | --help)
      sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      ;;
    *)
      log "ERROR: unknown option: $1"
      return 2
      ;;
  esac
}

main "$@"
