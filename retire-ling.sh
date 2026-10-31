#!/usr/bin/env bash
# Retire the unreliable Ling lane. Keep all weights, caches, images and API keys.
set -euo pipefail
if [ "$(id -u)" = 0 ] && [ "${ALLOW_ROOT:-0}" != 1 ]; then
  echo 'Run as the Spark account, not root.' >&2; exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$HOME/.config/qwen38"
. "$HERE/lanes.sh"
. "$HERE/vllm-lane.sh"
UNIT=qwen38-ling.service
INSTALLED="$LANES_UNIT_DIR/$UNIT"
VLANE=ling; VLANE_SHORT=Ling; CONTAINER=qwen38-ling
if [ -f "$INSTALLED" ]; then
  echo 'Retiring Ling: stopping its engine and removing its service; cached data stays.'
  sudo systemctl stop "$UNIT"
  docker rm -f qwen38-ling qwen38-ling-compile >/dev/null 2>&1 || true
  vlane_uninstall_unit
fi
vlane_uninstall_clients
